#!/usr/bin/env python3
"""Scaled hybrid-conversation seam collection v2 (freeze + collect through real Pi).

Mission: scale the ONE family that teaches the operator's exact conversational
seam patterns (greeting -> plain-text finish; chat opener flowing into tool
use; bash date -> correct weekday answer verified against the real clock;
small file-op conversations with verified outcomes; multi-step records where
a tool result is followed by more conversational Commentary before the
conversational finish) from 239 records / ~40K targets to thousands of
records, with the SAME verification bar as the v1 collection: every episode
is executed end-to-end through real Pi (scripts/e97_pi_native_tool_bridge.py
+ scripts/e97_pi_native_tool_transport.py), every tool result is a real
observation, every workspace outcome is oracle-checked, and nothing is
admitted to training.

Codec authority (unchanged from v1): scripts/e97_pi_native_codec.py. The
canonical episode ends at the first finish, and the only public assistant
text is Commentary (on tool turns) and finish.message, so the multi-turn
conversational structure is taught WITHIN episodes: conversational openers in
the user message, conversational Commentary between tool result and finish,
and conversational finish messages. This matches the frozen acceptance panel
(configs/pi/e97-e1-chat-probe-panel-v1.json), whose three cases are all
single-user-message episodes.

Environment-drift compat (documented in the collection STATUS.md): pi 0.87.0
drifted three web-tool description strings in its bundled web-access
extension, breaking the owner bridge's frozen-manifest tool contract. This
collector therefore drives Pi with the pinned provider extension
configs/pi/e97-pi-native-frozen-tools-v2.ts (byte-copy of the v1 provider
with request tools pinned to the transport config's frozen manifest specs).
All authority files from v1 stay byte-identical.

Commands:
  freeze   author the full case plan (deterministic phrasing banks; zero
           protected-panel or panel-literal content) and publish plan-private.json
  collect  execute every case through real Pi with checkpoint/resume: already
           verified episodes are re-encoded from their episode-private.json,
           rejections are kept, and interrupted runs resume where they left
           off. Writes candidate-authority + summary only when the full plan
           is attempted and the minimum verified count is reached.
"""
import argparse,datetime,hashlib,json,os,re,shutil,signal,subprocess,threading,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import tiktoken

from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_tool_bridge import NativePiToolBridge
from scripts.e97_pi_native_tool_transport import serve_pi_native_tools
from scripts.eval_e97_native_execution import publish,sha
from scripts.build_e97_pi_native_curriculum import (SYSTEM,MANIFEST_SHA,EXTENSIONS,grounded,action,finish,frame,encode_candidate,write_authority,last_result)

SEED='e97-hybrid-conversation-v2-20260922'
PLAN_SCHEMA='emender-e97-hybrid-conversation-plan-v2'
PROVIDER_V2=Path('configs/pi/e97-pi-native-frozen-tools-v2.ts')
WEEKDAYS=('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
WEEKDAY_FULL={'mon':'Monday','tue':'Tuesday','wed':'Wednesday','thu':'Thursday','fri':'Friday','sat':'Saturday','sun':'Sunday'}
MONTH_FULL={'jan':'January','feb':'February','mar':'March','apr':'April','may':'May','jun':'June','jul':'July','aug':'August','sep':'September','oct':'October','nov':'November','dec':'December'}
# Frozen acceptance-panel literals are evaluation entities for this family:
# they must never appear verbatim in any authored case (prompts, fixtures,
# finish values). Numeric literals are matched as standalone numbers so a
# five-digit port containing the digit sequence is not a false collision.
PANEL_PHRASES=('ledger_e1probe','Hi! I\'m just checking in','run the date command and tell me what day of the week','Open data/ledger_e1probe.txt','add its two operands with a shell command')
PANEL_NUMBERS=('157','284','441')
FORBIDDEN_PHRASES=('api key','api-key','password','secret','credentials','credential','purchase','production deploy','Reykjavik','weather_adapter','PI_NATIVE_MARKER','facts.txt')
FORBIDDEN_NUMBERS=('731','992')
_NUMBER_RES=tuple((re.compile(r'(?<![0-9A-Za-z])'+re.escape(n)+r'(?![0-9A-Za-z])'),n) for n in PANEL_NUMBERS+FORBIDDEN_NUMBERS)

def check_text(text):
 for needle in PANEL_PHRASES:
  if needle in text:raise ValueError(f'panel literal leakage: {needle!r}')
 for word in FORBIDDEN_PHRASES:
  if word in text:raise ValueError(f'forbidden content: {word!r}')
 for pattern,name in _NUMBER_RES:
  if pattern.search(text):raise ValueError(f'forbidden number: {name}')
 return text
def opaque(seed,n=16):return hashlib.sha256(seed.encode()).hexdigest()[:n]
def case_id(family,i):return f'pi-native-hybrid2-{family}-{i:05d}-{opaque(f"{SEED}:{family}:{i}",8)}'

# ---------------------------------------------------------------- date parsing
BARE_RE=re.compile(r'^([A-Za-z]{3}) ([A-Za-z]{3}) ([ 0-9][0-9]) (\d{2}:\d{2}:\d{2}) (\S+) (\d{4})$')
YMD_RE=re.compile(r'^(\d{4})-(\d{2})-(\d{2})$')
ABD_RE=re.compile(r'^([A-Za-z]+), ([A-Za-z]+) (\d{1,2}), (\d{4})$')
HM_RE=re.compile(r'^(\d{2}):(\d{2})$')

def parse_date_observation(text,fmt):
 """Deterministically parse a real date-command observation into template values."""
 s=text.strip()
 if fmt=='bare':
  m=BARE_RE.match(s)
  if not m:raise ValueError(f'unparsed bare date observation: {s!r}')
  abbr,mon,day,clock,zone,year=m.groups()
  weekday=WEEKDAY_FULL[abbr.lower()];month=MONTH_FULL[mon.lower()];day=int(day)
  return {'weekday':weekday,'month':month,'day':str(day),'year':year,'time':clock,'zone':zone,'date':f'{month} {day}, {year}','ymd':f'{year}-{mon.title()}-{day:02d}'}
 if fmt=='A':
  if s not in WEEKDAYS:raise ValueError(f'unparsed weekday observation: {s!r}')
  return {'weekday':s}
 if fmt=='A-ymd':
  parts=s.split(' ',1)
  if len(parts)!=2 or parts[0] not in WEEKDAYS or not YMD_RE.match(parts[1]):raise ValueError(f'unparsed weekday-date observation: {s!r}')
  return {'weekday':parts[0],'ymd':parts[1]}
 if fmt=='u-ymd':
  if not YMD_RE.match(s):raise ValueError(f'unparsed utc date observation: {s!r}')
  return {'ymd':s}
 if fmt=='u-HM':
  if not HM_RE.match(s):raise ValueError(f'unparsed utc time observation: {s!r}')
  return {'time':s}
 if fmt=='A-BdY':
  m=ABD_RE.match(s)
  if not m:raise ValueError(f'unparsed long-date observation: {s!r}')
  weekday,month,day,year=m.groups()
  if weekday not in WEEKDAYS:raise ValueError(f'bad weekday: {s!r}')
  return {'weekday':weekday,'month':month,'day':str(int(day)),'year':year,'date':f'{month} {int(day)}, {year}'}
 raise ValueError(f'unknown date format {fmt}')

DATE_COMMANDS={'bare':"date",'A':"date '+%A'",'A-ymd':"date '+%A %Y-%m-%d'",'u-ymd':"date -u '+%Y-%m-%d'",'u-HM':"date -u '+%H:%M'",'A-BdY':"date '+%A, %B %-d, %Y'"}
DATE_USER={
 'bare':("What day of the week is it?","Run date and tell me the day of the week.","Can you run the date command and tell me what day it is?","What day is it today?","What's today's weekday?","Check the clock and tell me what day of the week we're on.","What day of the week does today fall on?","Tell me what day it is, please.","Quick one — what day is it?","What weekday is it right now?"),
 'A':("Tell me today's weekday.","What day of the week is it, in plain words?","Just the weekday for today, please.","What weekday is it? Give me the full day name.","Say what day of the week it is.","What day of the week are we on?","Which weekday is it?","Name today's day of the week.","What's the weekday today?","Today's day of the week?"),
 'A-ymd':("What day of the week is it, and what's the date?","Tell me the weekday and today's date.","What day is it, and what's today's date?","Give me today's weekday and the date.","Which weekday is it and what's the date today?","Day of the week plus the date, please.","What's today's day and date?","Tell me the day and the date.","What weekday is it, and what's today's date?","Name the day and the date for today."),
 'u-ymd':("What is today's date in UTC? Answer in plain text.","What's today's date in UTC?","Tell me the current UTC date.","What's the date right now, UTC?","Give me today's date in UTC, please.","What's the current date in UTC?","Today's UTC date, please.","UTC date for today?","What date is it in UTC right now?","Tell me today's date, in UTC."),
 'u-HM':("What time is it in UTC right now?","Tell me the current UTC time.","What's the UTC time at the moment?","Give me the time in UTC, please.","What time is it (UTC)?","Current UTC time, please.","UTC time right now?","What's the time in UTC at the moment?","Tell me the UTC clock time.","What time is UTC showing right now?"),
 'A-BdY':("What day is it today — the full date, please.","Tell me today's day of the week and the full date.","What's today's date and day of the week?","Give me the weekday and the full date for today.","What day of the week is it, and what's the full date?","Today's full date and day, please.","Tell me the full date for today.","What's the complete date today — day of the week included?","Name today's weekday and full date.","What's today's date, spelled out fully?")}
DATE_DECOR=("I keep losing track of the days.","Scheduling something and can't remember.","Off-by-one question, but I want it exact.","Writing this into my log.","Double-checking my calendar.","I'd rather trust the clock than my memory.","Need it for my timesheet.","My planner says one thing; the clock wins.","This decides a deadline, so be precise.","My watch is probably wrong.","Planning tomorrow's work.","Just want the real answer, not a guess.")
DATE_TEMPLATES={
 'bare':("It's {weekday} — today is {date}.","Today is {weekday}, {date}.","It's {weekday}! The date is {date}.","That would be {weekday} — {date}.","Today falls on a {weekday}; the date is {date}.","It's {weekday}, {date} ({time}, {zone})."),
 'A':("It's {weekday} today.","Today is {weekday}.","It's a {weekday}.","The day of the week is {weekday}.","That would be {weekday}.","We're on {weekday}."),
 'A-ymd':("It's {weekday} — {ymd}.","Today is {weekday}, {ymd}.","It's {weekday}, and the date is {ymd}.","Today is {ymd}, a {weekday}."),
 'u-ymd':("Today's date in UTC is {ymd}.","It's {ymd} in UTC today.","The current UTC date is {ymd}.","UTC date today: {ymd}.","In UTC, today is {ymd}."),
 'u-HM':("It's currently {time} UTC.","The time in UTC is {time}.","It's {time} UTC right now.","UTC time is {time}.","Right now it's {time} in UTC."),
 'A-BdY':("It's {weekday}, {date}.","Today is {weekday}, {date}.","It's {weekday} — the full date is {date}.","Today is {date}, a {weekday}.")}
DATE_ANALYSES={
 'bare':("The current day and date are environment state I cannot know from the conversation; running the date command observes them. The output shows {weekday}, {date}, so the plain-text answer states the observed weekday and date.","I cannot know today's weekday from the prompt alone; the minimal tool to observe it is the date command. The observation shows it is {weekday}, {date}, and the user-facing answer states exactly that observed day.","The day of the week is a clock fact, not something in the conversation; running date grounds the answer in a real observation. The command observed {weekday} and {date}, so the plain-text answer is that observed weekday and date."),
 'A':("The weekday is environment state I cannot know from the prompt alone; the date command observes it directly. The output says it is {weekday}, so the plain-text answer states that observed day.","I cannot know today's day of the week without checking the clock; running date with a weekday-only format observes it. The observation is {weekday}, and the user-facing answer states exactly that.","The current weekday is not in the conversation; the minimal grounded move is to run the date command and read the observed weekday. The tool observed {weekday}, which the plain-text answer states."),
 'A-ymd':("Both the weekday and the date are clock state I cannot know from the prompt; one date command observes both. The output shows {weekday} and {ymd}, so the plain-text answer states both observed values.","The day and the date are environment state; running date with a combined format observes them in one call. The observation shows {weekday}, {ymd}, and the answer states exactly those observed values.","I need the real clock for both the weekday and the date; the date command is the minimal tool to observe them. It observed {weekday} and {ymd}, so the plain-text answer states both."),
 'u-ymd':("The current UTC date is environment state I cannot know from the prompt alone; the minimal tool to observe it is the UTC date command. The output says {ymd}, and the user-facing answer states exactly that observed date.","Today's UTC date is a clock fact, not conversation content; running date -u observes it directly. The observation is {ymd}, so the plain-text answer is that observed date.","The UTC date is not something I can recall; the grounded move is to observe it with the date command. The tool observed {ymd}, and the answer states that observed value."),
 'u-HM':("The current UTC time is environment state I cannot know from the prompt; the minimal tool to observe it is the UTC date command with a time format. The output says {time}, so the plain-text answer states that observed time.","The time of day is clock state; running date -u with a time format observes it. The observation is {time} UTC, and the user-facing answer states exactly that observed time.","I cannot know the current time from the conversation; the date command observes it. The tool observed {time} in UTC, which the plain-text answer states."),
 'A-BdY':("The full day and date are environment state I cannot know from the prompt; one date command with a long format observes both. The output shows {weekday}, {date}, so the plain-text answer states the observed weekday and full date.","The weekday and full date are clock facts; the grounded move is to run the date command and read what it actually says. It observed {weekday}, {date}, and the answer states exactly those observed values.","I need the real clock for the day and the full date; the date command is the minimal tool. The observation shows {weekday}, {date}, so the plain-text answer states both observed values.")}
DATE_COMMENTARIES={'bare':("Checking the clock for you — one sec.","Let me check what day it is.","Sure — checking today's date now.","One moment, checking the clock.","Happy to — let me look at the date."),
 'A':("Checking the day of the week.","One sec — checking the weekday.","Let me check what day it is.","Checking the clock for you."),
 'A-ymd':("Let me check the day and the date.","One sec — checking the clock.","Checking today's day and date."),
 'u-ymd':("Checking the current UTC date.","One sec — checking the UTC date.","Let me look at today's UTC date."),
 'u-HM':("Checking the current UTC time.","One sec — checking the clock.","Let me check the UTC time."),
 'A-BdY':("Checking the full date for you.","One sec — checking today's date.","Let me check the day and full date.")}
DATE_FINISH_COMMENTARIES={'bare':("Got it from the clock.","That's what the clock says.","Checked — here's the day.","The clock has the answer."),
 'A':("Got the day from the clock.","Checked the weekday.","That's the observed day."),
 'A-ymd':("Here's the day and date.","Checked both for you."),
 'u-ymd':("Checked the UTC date.","Here's the UTC date."),
 'u-HM':("Here's the current UTC time.","Checked the clock."),
 'A-BdY':("Here's the full date.","Checked the day and date.")}

def utc_family(fmt):return fmt.startswith('u-')
def weekday_now(utc):
 now=datetime.datetime.now(datetime.timezone.utc) if utc else datetime.datetime.now()
 return now.strftime('%A')

# ------------------------------------------------------------- phrasing banks
GREETINGS=(
 ("Hey!",'casual'),("Hey there!",'casual'),("Hey, what's up?",'casual'),("hey what's up buddy",'casual'),
 ("Yo! How's it going?",'casual'),("Hiya!",'casual'),("What's up?",'casual'),("Hey hey!",'casual'),
 ("Morning!",'casual'),("Hey, you around?",'casual'),("Hi hi!",'casual'),("Hey friend!",'casual'),
 ("Hello.",'direct'),("Hi.",'direct'),("Hi — are you around?",'direct'),("Hello! Quick check-in.",'direct'),
 ("Hey, checking in.",'direct'),("Hi, you there?",'direct'),("Hello, status check.",'direct'),("Hi! You free?",'direct'),
 ("Good afternoon!",'polite'),("Good morning!",'polite'),("Good evening!",'polite'),
 ("Hello, I hope you're doing well.",'polite'),("Hi there, hope all is well!",'polite'),
 ("Greetings!",'polite'),("Hello there.",'polite'),("Good day!",'polite'),("Hi, pleased to see you.",'polite'),
 ("Hey! Long time.",'casual'),("Hi again!",'casual'),("Hello again.",'polite'),("Hey, back again.",'casual'))
GREETING_CONTEXT=(
 "Just got my coffee going.","I'm back at my desk for the afternoon.","Long day here — taking a quick breather.",
 "Just wrapping up a few things.","Settling in for the evening.","About to dig into some work and wanted to say hi first.",
 "The morning's been quiet so far.","Taking a five-minute break.","Things are finally calming down over here.",
 "Just finished a big chunk of work.","Wanted to check in before I start on something.","Between tasks at the moment.",
 "Storm's rolling in outside my window.","My desk is finally clean.","Fresh pot of tea just finished brewing.",
 "Kicking off the week and saying hello.","Wrapping up my notes for the day.","The afternoon slump is hitting.",
 "First break of the day.","Quiet evening ahead.","Just rearranged my desk setup.")
GREETING_REPLIES={
 'casual':("Hey! Not much — just hanging out, ready to help. What can I do for you?",
  "Hey there! Doing great, thanks for asking. What's on your mind?",
  "What's up! All good on my end — what do you need?",
  "Hey buddy! All systems green over here. What can I help you with?",
  "Hey! Can't complain. What are you working on?",
  "Yo! Doing just fine — ready when you are.",
  "Hey hey! Good to hear from you. What's up?",
  "Not much! Just idling until you give me something to do. What do you need?",
  "Hey! All quiet on my end. What can I do for you?",
  "Hey there — doing well! What's on the docket today?"),
 'direct':("Hello. I'm here and ready — what do you need?",
  "Hi. Doing fine. What would you like me to do?",
  "Hello! Present and ready for a task.",
  "Hi — I'm here. What's the task?",
  "Hello. All good here. What do you need done?",
  "Hi! Standing by — what would you like?"),
 'polite':("Good afternoon! I'm doing very well, thank you for asking. How may I help you today?",
  "Hello! I'm quite well, thank you. What can I do for you?",
  "Good day to you! Everything is going smoothly on my end. How may I be of assistance?",
  "Hello there — I'm doing well, thank you kindly. What would you like help with today?",
  "Greetings! I'm very well, thank you. Please let me know what you need.",
  "Hello! Wonderful to hear from you. I'm ready to help whenever you are.")}
GREETING_ANALYSES=(
 "The user is greeting me and asking how I am. This is a conversational check-in with no task and no fact to look up; no tool is needed, and the right response is a plain-text friendly reply that answers the greeting and offers help.",
 "This is a pure greeting. There is nothing to observe in the environment — no file, no clock fact, no computation — so calling any tool would be wrong. The correct move is to answer conversationally in plain text.",
 "The opener is smalltalk, not a task. The decision the seam requires is recognizing that no tool is needed here: the answer comes from the conversation itself, as a friendly plain-text reply.",
 "A greeting deserves a greeting. There is no observation to ground and no question about the workspace, so the plain-text finish path is correct and any tool call would be a mistake.",
 "The user said hello and asked how things are going. That is chat, not a task — the correct behavior is the no-tool plain-text answer that greets back and invites the next request.")
SMALLTALK_USER=(
 ("Thanks for the earlier help!",'thanks'),("Appreciate you being here.",'thanks'),("Just wanted to say thanks — you've been handy.",'thanks'),
 ("What can you actually do around here?",'capability'),("What tools do you have?",'capability'),("What kinds of things can you help with?",'capability'),
 ("How do you usually get things done?",'capability'),("What's your setup — what can I ask you for?",'capability'),
 ("Working hard or hardly working?",'banter'),("Doing anything fun over there?",'banter'),("You must be busy, huh?",'banter'),
 ("Bet you don't get many easy days.",'banter'),("Slow day today?",'banter'),("Anything interesting on your end?",'banter'),
 ("Just saying hi before I start my task.",'chat'),("Warming up before I ask you something.",'chat'),("I'll have a real question in a minute.",'chat'),
 ("Taking a breather — be right back with a task.",'chat'),("Getting organized before we dig in.",'chat'),("Hi! Just checking that you're responsive.",'chat'),
 ("Ping — are you awake?",'chat'),("You free for a quick question soon?",'chat'),("Ready when you are!",'chat'),("Just making sure the line is open.",'chat'))
SMALLTALK_REPLIES={
 'thanks':("You're welcome! Happy to help — what's next?","Anytime! That's what I'm here for.","Glad it helped! What else can I do for you?","You're most welcome. Ready for the next thing whenever you are."),
 'capability':("I can read and edit files, run shell commands, and answer questions from what I actually observe. Point me at a task!","On this machine I can read and edit workspace files and run bash commands, then report what I really saw. What would you like?","I work with files and shell commands — I check real outputs before answering. What can I get you?","I can look at your files, make small edits, and run commands, always grounding answers in what I observe. What's the job?"),
 'banter':("Ha! I keep busy enough — always something to read or run. What can I do for you?","Hardly a dull moment over here. What's up?","Working hard, as always — the tools don't run themselves. What do you need?","I stay pretty busy between your questions! What's next?"),
 'chat':("Hi! I'm here and ready. Take your time.","Hello! Line's open — what do you need?","Hey! I'm listening. Fire away when ready.","All set on my end — what's the task?")}
SMALLTALK_ANALYSES=(
 "The user is making small talk rather than assigning a task. There is no fact to observe and no file to touch, so no tool is needed; the correct response is a short, friendly plain-text reply.",
 "This is conversational filler — thanks, banter, or a readiness check. The seam decision is chat-versus-tool: this is chat, so the answer is plain text with no tool call.",
 "Nothing in this message needs the environment: it is talk, not a task. The right behavior is a conversational plain-text reply and no tool use.",
 "The message is social, not operational. No observation is required, so I answer directly in plain text and leave the tools alone.")
CHAT_OPENERS=(
 ("casual","Hey what's up buddy — while you're here, could you "),("casual","Hey! Quick thing: could you "),
 ("casual","Yo, can you "),("casual","Hey hey — would you "),("casual","Sup! While I've got you, could you "),
 ("casual","Hey, got a sec? Could you "),("casual","Hiya! Could you "),("casual","Hey friend, can you "),
 ("casual","What's up! Also, can you "),("casual","Hey there — while you're around, please "),
 ("direct","Hello. Please "),("direct","Hi — task for you: please "),("direct","Hello, please "),
 ("direct","Okay, straight to it: "),("direct","Hi. Go ahead and "),("direct","New task: please "),
 ("direct","Hello! Next up — "),("direct","Checking in with a task: please "),
 ("polite","Good afternoon! When you have a moment, could you please "),("polite","Hi there! If you don't mind, please "),
 ("polite","Hello! Whenever you're ready, could you kindly "),("polite","Good day — I'd appreciate it if you could "),
 ("polite","Hi! Hope you're well. Could you please "),("polite","Hello again! Would you mind "))
def seam_prompt(i,clause):
 register,prefix=pick(CHAT_OPENERS,i)
 return prefix+clause+('?' if register in ('casual','polite') else '.')
PRE_TOOL_COMMENTARY=(
 "Sure thing — let me check that for you.","On it — checking now.","Happy to! One sec.",
 "Let me take a look.","You got it — checking that now.","Sure, let me grab that.",
 "Absolutely — one moment.","Right away. Let me check.","Of course — checking that for you now.",
 "Consider it done — verifying first.","Easy enough. Let me look.","Let me go check that.",
 "Great question — let me verify it properly.","I'll check the real thing for you.","One quick check first.")
MID_CONVERSATIONAL_COMMENTARY=(
 "Nice, got it — and there's more to check.","That answers part of it; let me finish the job.",
 "Good, that's observed. One more thing to grab.","Alright, got that — now for the rest.",
 "Halfway there. Checking the other half now.","That's useful — let me also confirm the other value.",
 "Got the first piece. Now the second.","Spotted it — one more observation to go.",
 "That looks good so far. Finishing the check now.","Okay, one more look and I'll wrap this up.")
FINISH_CONVERSATIONAL_COMMENTARY=(
 "There you go!","Here's your answer.","All set — here it is.","Got it.","Done — short answer:",
 "That's it!","Here you go.","Wrapped up — here's the result.","Checked and confirmed.","Done checking.")
READ_FACT_ANALYSES=(
 "The user is asking about a fact recorded in the workspace file {path}; reading it observes the value directly instead of guessing. The file records {key} = {value}, so the plain-text answer is exactly that observed value.",
 "The fact lives in {path}, not in the conversation, so the minimal grounded move is to read that file. The read observed {key} = {value}, and the user-facing answer states that observed value.",
 "This is a workspace fact question; the correct seam behavior is to observe the file rather than answer from memory. Reading {path} shows {key} = {value}, which is the plain-text answer.",
 "The value is recorded in {path}; I cannot know it without reading. The read observes {key} = {value}, so the answer is exactly what the file says.")
COUNT_ANALYSES=(
 "The line count of {path} is a fact about the workspace, and counting by eye is error-prone; running wc -l observes it. The command observed {count} lines, so the plain-text answer is that observed count.",
 "How many lines a file has is environment state; the minimal tool to observe it is wc -l on {path}. The command returned {count}, and the answer states exactly that observed number.",
 "The count lives in the real file, not in the conversation; running wc -l grounds the answer. The tool observed {count} lines in {path}, which is the plain-text answer.")
DERIVE_ANALYSES=(
 "The sum is a derived fact, so both operands must be observed first. Reading {path} observed a = {a} and b = {b}; computing their sum with the shell makes the result an observed fact instead of mental arithmetic. The command observed {sum}, so the plain-text answer is that observed sum.",
 "This needs two observations and one computation: read {path} for the operands, then add them with a shell command. The read observed a = {a} and b = {b}, the shell observed {sum}, and the answer states exactly that observed sum.",
 "Arithmetic from memory is not grounded; the operands are in {path} and the sum should come from the shell. Reading observed a = {a} and b = {b}, and computing with bash observed {sum} — the plain-text answer is that observed result.")
WRITE_ANALYSES=(
 "The user wants a specific new file; writing it and reading it back verifies the outcome. The write put exactly {content} into {path}, the read-back observed the same content, so the task is complete and the answer is done.",
 "Creating {path} is an action with a verifiable outcome: write the exact content, then read it back to confirm. The read-back observed {content}, which confirms the write, so the plain-text answer is done.",
 "The right pattern for a write task is write-then-verify: create {path} with the requested content and read it back. The observed read-back matches what was requested, so the answer is done.")
EDIT_ANALYSES=(
 "The user wants a value changed in {path}; the edit is an exact replacement and the read-back verifies it. The edit replaced {old} with {new}, the read-back observed {new}, so the outcome is verified and the answer is done.",
 "Editing requires the exact old text and the new text; after replacing {old} with {new} in {path}, reading it back confirms the file now holds {new}. The outcome is observed, so the plain-text answer is done.",
 "This is a small file edit with a verifiable outcome: replace {old} with {new} in {path}, then read the file to confirm. The read-back shows the change landed, so the task is complete.")
def step_finish_static(message,analysis,commentary,requires):return grounded(finish(message),analysis,commentary,requires)

# ------------------------------------------------------------ case generators
def register_of(i):return ('casual','direct','polite')[i%3]
def pick(bank,i):return bank[i%len(bank)]

def base_case(i,family,prompt,steps,files=None,pure_chat=False,answer_must_be_observed=False,expected_answer=None,answer_is_observation=False,date_fmt=None):
 check_text(prompt)
 for step in steps:
  check_text(json.dumps(step.get('arguments',{}))) if isinstance(step.get('arguments'),dict) else None
 for path,content in (files or {}).items():
  if path.startswith('/') or '..' in path.split('/'):raise ValueError('fixture path')
  check_text(path);check_text(content)
 return dict(id=case_id(family,i),category='hybrid',family=family,prompt=prompt,files=files or {},expected_files=dict(files or {}),steps=steps,supervise_from=0,requires_error=False,repository_discovery=False,pure_chat=pure_chat,answer_must_be_observed=answer_must_be_observed,expected_answer=expected_answer,answer_is_observation=answer_is_observation,date_fmt=date_fmt)

def greeting_case(i):
 text,register=pick(GREETINGS,i)
 if i>=len(GREETINGS):
  text=text+' '+GREETING_CONTEXT[(i-len(GREETINGS))//len(GREETINGS)]
 reply=pick(GREETING_REPLIES[register_of(i)],i//3);analysis=pick(GREETING_ANALYSES,i)
 return base_case(i,'hybrid-greeting',text,[step_finish_static(reply,analysis,None,[])],pure_chat=True)

def smalltalk_case(i):
 text,kind=pick(SMALLTALK_USER,i)
 if i>=len(SMALLTALK_USER):
  text=text+' '+GREETING_CONTEXT[(i-len(SMALLTALK_USER))//len(SMALLTALK_USER)]
 reply=pick(SMALLTALK_REPLIES[kind],i);analysis=pick(SMALLTALK_ANALYSES,i)
 return base_case(i,'hybrid-smalltalk',text,[step_finish_static(reply,analysis,None,[])],pure_chat=True)

def pure_chat_fact_case(i,kind=None):
 cid=opaque(f'{SEED}:purechat:{i}',16);kind=i%3 if kind is None else kind
 if kind==0:  # supplied-fact restatement
  value='RELIC_'+opaque(cid,12)
  prompt=pick((f'The workspace ledger records the value {value}. What value does the ledger record? Answer in plain text without using any tools.',
   f'Heads up: the log notes the code {value}. What code do the notes record? Answer plainly — no tools needed.',
   f'The manifest entry says the value is {value}. Without tools, what value does the manifest record?',
   f'For your reference, the recorded tag is {value}. What is the recorded tag? Answer in plain text, no tools.',
   f'The registry lists the label {value}. Tell me the label in plain text without using any tools.'),i)
  analysis=f'The fact is supplied in the conversation: the recorded value is {value}. No tool is needed because the value is already given here; the plain-text answer is exactly {value}.'
  answer=value;family='hybrid-pure-chat-supplied'
 elif kind==1:  # supplied-label selection
  alpha='CHOSEN_'+opaque(cid+':a',10);beta='SHUNTED_'+opaque(cid+':b',10)
  prompt=pick((f'For ticket {opaque(cid,8)}, the approved label is {alpha} and the draft label is {beta}. Return only the approved label in plain text without using any tools.',
   f'The pinned name is {alpha} and the proposed name is {beta}. Which name is pinned? Answer plainly, no tools.',
   f'Between {alpha} (current) and {beta} (old), which one is current? Reply in plain text without tools.',
   f'The active tag is {alpha}; the retired tag is {beta}. Tell me the active tag in plain text, no tools needed.'),i)
  analysis=f'Both labels are supplied in the conversation; the correct one is {alpha}, so no tool is needed and the plain-text answer is exactly that label.'
  answer=alpha;family='hybrid-pure-chat-selection'
 else:  # supplied-value contrast
  a=100+(i*13)%900;b=100+(i*29)%900
  while b==a or a in (157,284,441,731,992) or b in (157,284,441,731,992) or (a+b) in (157,284,441,731,992):
   a=(a+17)%900+100;b=(b+29)%900+40
  bigger=a>b
  prompt=pick((f'Value A is {a} and value B is {b}. Without tools, which value is larger? Answer plainly.',
   f'A is {a}, B is {b}. No tools: which is the bigger one? Plain text, please.',
   f'Given {a} and {b}, which number is larger? Answer in plain text without using any tools.',
   f'Compare for me — {a} versus {b}. Which is larger? Answer plainly, no tools needed.'),i) if bigger else pick((f'Value A is {a} and value B is {b}. Without tools, which value is smaller? Answer plainly.',
   f'A is {a}, B is {b}. No tools: which is the smaller one? Plain text, please.',
   f'Given {a} and {b}, which number is smaller? Answer in plain text without using any tools.',
   f'Compare for me — {a} versus {b}. Which is smaller? Answer plainly, no tools needed.'),i)
  picked=str(max(a,b)) if bigger else str(min(a,b))
  analysis=f'Both values are supplied in the conversation: A is {a} and B is {b}. No tool is needed because the comparison is between two given numbers; the plain-text answer is {picked}.'
  answer=picked;family='hybrid-pure-chat-contrast'
 commentary=f'The answer is {answer}.'
 return base_case(i,family,prompt,[step_finish_static(answer,analysis,commentary,[answer])],pure_chat=True)

def date_command_step(fmt,i):
 commentary=pick(DATE_COMMENTARIES[fmt],i)
 return grounded(action('bash',{'command':DATE_COMMANDS[fmt]}),
  f'The current date and time are environment state I cannot know from the prompt alone; the minimal tool to observe them is the date command.',
  'Checking the clock for you.',[])

def date_case(i):
 fmts=('bare','A','A-ymd','u-ymd','u-HM','A-BdY')
 fmt=fmts[i%len(fmts)];k=i//len(fmts)
 phrases=DATE_USER[fmt]
 prompt=phrases[k] if k<len(phrases) else phrases[k%len(phrases)]+' '+DATE_DECOR[(k//len(phrases)-1)%len(DATE_DECOR)]
 exact=(k//len(fmts))%8==0  # a slice keeps the terse exact-observation path
 if exact and fmt in ('A','u-ymd','u-HM'):
  answer={'name':'finish','dynamic':'observation-exact','analysis_dynamic':'observation','commentary_dynamic':'observation'}
  steps=[date_command_step(fmt,i),answer]
  return base_case(i,'hybrid-date-question',prompt,steps,date_fmt=fmt,answer_is_observation=True)
 template=pick(DATE_TEMPLATES[fmt],i);analysis=pick(DATE_ANALYSES[fmt],i);commentary=pick(DATE_FINISH_COMMENTARIES[fmt],i)
 finish_step={'name':'finish','dynamic':'observation-date','fmt':fmt,'template':template,'analysis_template':analysis,'commentary':commentary}
 steps=[date_command_step(fmt,i),finish_step]
 return base_case(i,'hybrid-date-question',prompt,steps,date_fmt=fmt)

DATE_SEAM_CLAUSES={
 'bare':("run date and tell me the day of the week","run the date command and tell me what day it is","check the clock and tell me what day of the week we're on","tell me what day it is today","run date and tell me today's weekday","check what day of the week it is"),
 'A':("tell me today's weekday","tell me what day of the week it is, in plain words","give me the full day name for today","tell me the day of the week"),
 'A-ymd':("tell me the day of the week and today's date","run date and give me the weekday plus the date","tell me what day it is and what the date is"),
 'u-ymd':("tell me today's date in UTC","check the current UTC date for me","run date and tell me today's UTC date"),
 'A-BdY':("tell me what day it is — the full date, please","run date and give me the weekday and the full date","tell me today's day of the week and the full date")}
SEAM_TAILS=('',' No rush.',' Thanks!')
def seam_date_case(i):
 fmts=('bare','A','A-ymd','u-ymd','A-BdY')
 fmt=fmts[i%len(fmts)]
 prompt=seam_prompt(i,pick(DATE_SEAM_CLAUSES[fmt],i))+SEAM_TAILS[(i//100)%len(SEAM_TAILS)]
 template=pick(DATE_TEMPLATES[fmt],i);analysis=pick(DATE_ANALYSES[fmt],i);commentary=pick(DATE_FINISH_COMMENTARIES[fmt],i)
 ack=pick(PRE_TOOL_COMMENTARY,i)
 first=grounded(action('bash',{'command':DATE_COMMANDS[fmt]}),
  f'The user opened conversationally and asked for the current day or date, which is environment state I cannot know from the prompt; the minimal tool to observe it is the date command.',
  ack,[])
 finish_step={'name':'finish','dynamic':'observation-date','fmt':fmt,'template':template,'analysis_template':analysis,'commentary':commentary}
 return base_case(i,'hybrid-chat-into-tool-date',prompt,[first,finish_step],date_fmt=fmt)

def read_fact_case(i,seam=False):
 cid=opaque(f'{SEED}:read:{i}:{int(bool(seam))}',16);tag=opaque(cid,10)
 kinds=('port','release','version','owner','token')
 kind=kinds[i%len(kinds)]
 path={'port':f'config/service_{tag}.ini','release':f'docs/notes_{tag}.txt','version':f'state/build_{tag}.txt','owner':f'data/owner_{tag}.txt','token':f'data/token_{tag}.txt'}[kind]
 value={'port':str(10000+(i*757)%48000),'release':'REL_'+opaque(cid,14),'version':'v'+opaque(cid,6),'owner':opaque(cid,8),'token':'TKN_'+opaque(cid,10)}[kind]
 key={'port':'port','release':'release','version':'version','owner':'owner','token':'token'}[kind]
 files={path:f'{key} = {value}\n'}
 ask_banks={
 'port':(f'What port does the config at {path} declare? Answer with only the port number in plain text.',f'Check the config at {path} and tell me the declared port.',f'Which port is set in {path}? Just the number, please.',f'Look at {path} and report the port it declares.'),
 'release':(f'What release marker does {path} record? Answer with only the marker in plain text.',f'Check {path} and tell me the release marker.',f'Which release is recorded in {path}? Just the marker.',f'Look at {path} and give me the release marker.'),
 'version':(f'What version does {path} list? Answer with only the version in plain text.',f'Check {path} and tell me the version it lists.',f'Which version is recorded in {path}? Just the version.',f'Look at {path} and report the recorded version.'),
 'owner':(f'Who is the owner recorded in {path}? Answer with only the owner in plain text.',f'Check {path} and tell me who the recorded owner is.',f'Which owner is listed in {path}? Just the owner.',f'Look at {path} and give me the recorded owner.'),
 'token':(f'What token does {path} store? Answer with only the token in plain text.',f'Check {path} and tell me the stored token.',f'Which token is kept in {path}? Just the token.',f'Look at {path} and report the token it stores.')}
 ask=pick(ask_banks[kind],i)
 conversational=i%2==0
 if conversational:
  ask={'port':f'Hey, can you check the config at {path} and tell me what port it declares?',
   'release':f'Could you look at {path} and tell me the release marker?',
   'version':f'What version are we on? It should be recorded in {path}.',
   'owner':f'Who owns this thing? The answer is in {path}.',
   'token':f'Can you dig up the token stored at {path} for me?'}[kind]
 if seam:
  prompt=seam_prompt(i,{'port':f'check the config at {path} and tell me what port it declares',
   'release':f'look at {path} and tell me the release marker',
   'version':f'check {path} and tell me what version it lists',
   'owner':f'look in {path} and tell me who the recorded owner is',
   'token':f'dig up the token stored at {path}'}[kind])
 else:
  prompt=ask
 first=grounded(action('read',{'path':path}),pick(READ_FACT_ANALYSES,i).format(path=path,key=key,value=value),pick(PRE_TOOL_COMMENTARY,i),[value])
 if conversational:
  message=pick((f'There you go — the {key} is {value}.',f'Checked it: the {key} is {value}.',f'Found it — {value}.',f'Looked at {path} and the {key} is {value}.'),i)
  commentary=pick(FINISH_CONVERSATIONAL_COMMENTARY,i)
 else:
  message=value;commentary=f'The {key} is {value}.'
 second=step_finish_static(message,first['analysis'],commentary,[value])
 family='hybrid-chat-into-tool-read' if seam else 'hybrid-read-fact'
 return base_case(i,family,prompt,[first,second],files=files,answer_must_be_observed=True,expected_answer=value)

def count_case(i,seam=False):
 cid=opaque(f'{SEED}:count:{i}:{int(bool(seam))}',16);tag=opaque(cid,10);count=3+(i*7)%40
 path=f'notes/list_{tag}.txt'
 files={path:''.join(f'entry {j}: {opaque(cid+str(j),6)}\n' for j in range(1,count+1))}
 ask=f'How many lines does {path} have? Answer with only the number in plain text.'
 if seam:
  prompt=seam_prompt(i,f'check how many lines are in {path}')
 else:
  prompt=ask if i%2 else f'Hey, quick count: how many lines does {path} have?'
 first=grounded(action('bash',{'command':f'wc -l < {path}'}),pick(COUNT_ANALYSES,i).format(path=path,count=count),pick(PRE_TOOL_COMMENTARY,i),[str(count)])
 conversational=i%2==0
 message=(f'There are {count} lines in {path}.' if conversational else str(count))
 commentary=(pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if conversational else f'The file has {count} lines.')
 second=step_finish_static(message,f'The command observed {count} lines; the plain-text answer is exactly that observed count.',commentary,[str(count)])
 family='hybrid-chat-into-tool-count' if seam else 'hybrid-bash-count'
 return base_case(i,family,prompt,[first,second],files=files,answer_must_be_observed=True,expected_answer=str(count))

def derive_case(i,seam=False):
 cid=opaque(f'{SEED}:derive:{i}:{int(bool(seam))}',16);tag=opaque(cid,10);a=101+(i*17)%850;b=42+(i*23)%850
 while b in (157,284,441,731,992) or a in (157,284,441,731,992) or a+b in (157,284,441,731,992):
  a=(a+13)%900+100;b=(b+29)%900+40
 total=a+b
 path=f'data/ledger_{tag}.txt'
 files={path:f'a = {a}\nb = {b}\n'}
 ask=f'Open {path}, add its two operands, and answer with only the sum in plain text.'
 if seam:
  prompt=seam_prompt(i,f'open {path}, add the two operands, and tell me the sum')
 else:
  prompt=ask if i%2 else f'Can you grab the two numbers in {path} and add them up for me? Answer with just the sum.'
 first=grounded(action('read',{'path':path}),
  f'The user asks for a derived fact; reading {path} observes the two operands before any arithmetic is trusted.',
  pick(PRE_TOOL_COMMENTARY,i),[f'ledger_{tag}'])
 second=grounded(action('bash',{'command':f'printf \'%s\\n\' $(({a}+{b}))'}),
  pick(DERIVE_ANALYSES,i).format(path=path,a=a,b=b,sum=total),'Adding the observed operands now.',[str(a),str(b)])
 conversational=i%2==0
 message=(f'The sum is {total}.' if conversational else str(total))
 commentary=(pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if conversational else f'The sum is {total}.')
 third=step_finish_static(message,f'The command observed {total}; the plain-text answer is exactly that observed sum.',commentary,[str(total)])
 family='hybrid-chat-into-tool-derive' if seam else 'hybrid-bash-derive'
 return base_case(i,family,prompt,[first,second,third],files=files,answer_must_be_observed=True,expected_answer=str(total))

def write_case(i,seam=False):
 cid=opaque(f'{SEED}:write:{i}:{int(bool(seam))}',16);tag=opaque(cid,10);content='note-'+opaque(cid,14)
 path=f'out/note_{tag}.txt'
 files={};want={path:content+'\n'}
 ask=f'Create {path} with exactly {content} followed by a newline, read it back, and finish with exactly done.'
 if seam:
  prompt=seam_prompt(i,f'create {path} with exactly {content} followed by a newline, read it back, and confirm with done')
 else:
  prompt=ask if i%2 else f'Hey, make a new file at {path} containing exactly {content} (with a trailing newline), then read it back and confirm with done.'
 first=grounded(action('write',{'path':path,'content':content+'\n'}),pick(WRITE_ANALYSES,i).format(path=path,content=content),pick(PRE_TOOL_COMMENTARY,i),[])
 second=grounded(action('read',{'path':path}),'Reading the file back to verify the write landed exactly as requested.','Verifying the write.',[])
 conversational=i%2==0
 message=('done — created and verified.' if conversational else 'done')
 commentary=(pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if conversational else 'Verified.')
 third=step_finish_static(message,f'The read-back observed {content}, which confirms the write; the outcome is verified, so the plain-text answer is done.',commentary,[content])
 family='hybrid-chat-into-tool-write' if seam else 'hybrid-write-verify'
 return base_case(i,family,prompt,[first,second,third],files=files,answer_must_be_observed=True,expected_answer=content)

def edit_case(i,seam=False):
 cid=opaque(f'{SEED}:edit:{i}:{int(bool(seam))}',16);tag=opaque(cid,10);old='alpha';new='beta'
 path=f'config/mode_{tag}.txt'
 files={path:f'mode = {old}\nzone = {opaque(cid,4)}\n'};want=dict(files);want[path]=f'mode = {new}\nzone = '+files[path].split('zone = ')[1]
 ask=f'In {path}, change mode = {old} to mode = {new}, read it back, and finish with exactly done.'
 if seam:
  prompt=seam_prompt(i,f'edit {path} so mode = {old} becomes mode = {new}, read it back, and confirm with done')
 else:
  prompt=ask if i%2 else f'Quick edit: in {path}, flip mode from {old} to {new}, verify with a read, and finish with done.'
 first=grounded(action('edit',{'path':path,'edits':[{'oldText':f'mode = {old}','newText':f'mode = {new}'}]}),pick(EDIT_ANALYSES,i).format(path=path,old=f'mode = {old}',new=f'mode = {new}'),pick(PRE_TOOL_COMMENTARY,i),[])
 second=grounded(action('read',{'path':path}),'Reading the file back to verify the edit landed exactly as requested.','Verifying the edit.',[])
 conversational=i%2==0
 message=('done — mode is '+new+' now.' if conversational else 'done')
 commentary=(pick(FINISH_CONVERSATIONAL_COMMENTARY,i) if conversational else 'Verified.')
 third=step_finish_static(message,f'The read-back observed mode = {new}, which confirms the edit; the outcome is verified, so the plain-text answer is done.',commentary,[f'mode = {new}'])
 family='hybrid-chat-into-tool-edit' if seam else 'hybrid-edit-verify'
 return base_case(i,family,prompt,[first,second,third],files=files,answer_must_be_observed=True,expected_answer=f'mode = {new}')

def seam_chat_case(i):
 """Greeting-style opener carrying a supplied fact: chat flows, then finishes in plain text."""
 cid=opaque(f'{SEED}:seamchat:{i}',16);value='SUPPLIED_'+opaque(cid,12)
 prompt=seam_prompt(i,f'note that the access code is {value} and tell me what it is — answer plainly, no tools needed')
 analysis=f'The user greeted me and supplied the fact directly in the conversation: the access code is {value}. No tool is needed because the fact is already given here; the plain-text answer is exactly {value}.'
 message=pick((f'The access code is {value}.',f'Got it — {value}.',f'You said it yourself: {value}.',f'Noted! The access code is {value}.'),i)
 return base_case(i,'hybrid-chat-into-chat',prompt,[step_finish_static(message,analysis,None,[value])],pure_chat=True)

def tool_then_chat_case(i):
 """Multi-step: a tool result, then MORE conversational Commentary on the next assistant turn, then a conversational finish."""
 cid=opaque(f'{SEED}:tchat:{i}',16);tag=opaque(cid,10);kind=i%3
 if kind==0:  # two config facts
  p1=f'config/alpha_{tag}.ini';p2=f'config/beta_{tag}.ini'
  v1=str(10000+(i*757)%48000);v2=str(10000+(i*953)%48000)
  files={p1:f'port = {v1}\n',p2:f'port = {v2}\n'}
  prompt=f'I need two port numbers: the one in {p1} and the one in {p2}. Can you read both and tell me them together?'
  first=grounded(action('read',{'path':p1}),f'The user needs two facts, each in its own file; observing the first grounds the first half. The read observed port = {v1}.','Sure — checking both files now.',[v1])
  second=grounded(action('read',{'path':p2}),f'The first observation gave port = {v1}; the second fact still needs its own observation before answering. Reading {p2} observes the second port directly.',pick(MID_CONVERSATIONAL_COMMENTARY,i),[v1])
  message=f'The port in {p1} is {v1}, and the port in {p2} is {v2}.';requires=[v1,v2]
 elif kind==1:  # count then derive
  path=f'data/ledger_{tag}.txt';a=120+(i*11)%800;b=53+(i*31)%800;total=a+b
  while a in (157,284,441,731,992) or b in (157,284,441,731,992) or total in (157,284,441,731,992):a=(a+17)%900+100;b=(b+37)%900+50;total=a+b
  files={path:f'a = {a}\nb = {b}\n'}
  prompt=f'Open {path}, add the two operands with a shell command, and also tell me the line count of {path}. Take your time.'
  first=grounded(action('read',{'path':path}),f'The user wants a derived sum and a line count from {path}; observing the operands first grounds everything that follows. The read observed a = {a} and b = {b}.','Happy to — let me look at that file.',[f'ledger_{tag}'])
  second=grounded(action('bash',{'command':f'wc -l < {path} && printf \'%s\\n\' $(({a}+{b}))'}),f'The operands are observed (a = {a}, b = {b}); now the shell observes both the line count and the sum so both answers are grounded facts.',pick(MID_CONVERSATIONAL_COMMENTARY,i),[str(a),str(b)])
  message=f'The sum is {total} and {path} has 2 lines.';requires=[str(total)]
 else:  # read fact then write summary note
  src=f'data/spec_{tag}.txt';value='SPEC_'+opaque(cid,12);out=f'out/summary_{tag}.txt'
  files={src:f'marker = {value}\n'}
  prompt=f'Read the marker at {src}, then write a one-line note at {out} that contains the marker, read the note back, and tell me the marker.'
  first=grounded(action('read',{'path':src}),f'The marker is recorded in {src}; reading it observes the fact before anything is written. The read observed marker = {value}.','Let me read that first.',[value])
  second=grounded(action('write',{'path':out,'content':f'marker = {value}\n'}),f'The marker {value} is now observed; writing it into {out} and reading back verifies the requested note.',pick(MID_CONVERSATIONAL_COMMENTARY,i),[])
  third=grounded(action('read',{'path':out}),'Reading the note back to verify the write landed exactly as requested.','Verifying the note.',[])
  want={src:files[src],out:f'marker = {value}\n'}
  message=f'Done — the marker is {value} and the note is written.'
  closing=f'The marker {value} was observed from {src}, written into {out}, and the read-back confirmed it; every step is verified by a real observation, so the conversational finish reports the marker.'
  case=base_case(i,'hybrid-tool-then-chat',prompt,[first,second,third,step_finish_static(message,closing,pick(FINISH_CONVERSATIONAL_COMMENTARY,i),[value])],files=files,answer_must_be_observed=True,expected_answer=value)
  case['expected_files']=want
  return case
 closing=(f'The first read observed port = {v1} and the second read observed port = {v2}; with both facts actually observed, the user-facing answer is the conversational plain-text statement of both ports.'
  if kind==0 else
  f'The shell observed the line count and the sum {total} for real; with every requested fact grounded in an actual observation, the plain-text answer states the observed results.')
 final=step_finish_static(message,closing,pick(FINISH_CONVERSATIONAL_COMMENTARY,i),requires)
 return base_case(i,'hybrid-tool-then-chat',prompt,[first,second,final],files=files,answer_must_be_observed=True,expected_answer=requires[0])

# ---------------------------------------------------------------- distribution
DISTRIBUTION=(
 ('hybrid-greeting',650,greeting_case),
 ('hybrid-smalltalk',330,smalltalk_case),
 ('hybrid-pure-chat-supplied',140,lambda i:pure_chat_fact_case(i,kind=0)),
 ('hybrid-pure-chat-selection',130,lambda i:pure_chat_fact_case(i,kind=1)),
 ('hybrid-pure-chat-contrast',130,lambda i:pure_chat_fact_case(i,kind=2)),
 ('hybrid-date-question',750,date_case),
 ('hybrid-chat-into-tool-date',300,seam_date_case),
 ('hybrid-chat-into-tool-read',300,lambda i:read_fact_case(i,seam=True)),
 ('hybrid-chat-into-tool-derive',180,lambda i:derive_case(i,seam=True)),
 ('hybrid-chat-into-tool-count',100,lambda i:count_case(i,seam=True)),
 ('hybrid-chat-into-tool-write',30,lambda i:write_case(i,seam=True)),
 ('hybrid-chat-into-tool-edit',30,lambda i:edit_case(i,seam=True)),
 ('hybrid-chat-into-chat',200,seam_chat_case),
 ('hybrid-read-fact',330,lambda i:read_fact_case(i,seam=False)),
 ('hybrid-bash-count',180,lambda i:count_case(i,seam=False)),
 ('hybrid-bash-derive',330,lambda i:derive_case(i,seam=False)),
 ('hybrid-write-verify',120,lambda i:write_case(i,seam=False)),
 ('hybrid-edit-verify',120,lambda i:edit_case(i,seam=False)),
 ('hybrid-tool-then-chat',850,tool_then_chat_case))

def authority_files_v2():
 paths=[Path('scripts/build_e97_pi_native_curriculum.py'),Path(__file__),PROVIDER_V2,
  Path('configs/pi/e97-pi-native.ts'),Path('scripts/e97_pi_native_codec.py'),
  Path('scripts/e97_pi_native_tool_bridge.py'),Path('scripts/e97_pi_native_tool_transport.py'),*EXTENSIONS]
 return {str(p.resolve()):sha(p) for p in paths}

def make_all_cases():
 cases=[];distribution={}
 for family,count,fn in DISTRIBUTION:
  start=len(cases)
  for j in range(count):
   case=fn(j)
   if case['family']!=family:raise ValueError(f'family mismatch {case["family"]}!={family}')
   cases.append(case)
  distribution[family]=count
 if len({c['id'] for c in cases})!=len(cases):raise ValueError('case identity')
 if len({c['prompt'] for c in cases})!=len(cases):raise ValueError('case prompt uniqueness')
 # Static-step analysis grounding: every analysis_requires literal must appear in its analysis.
 for c in cases:
  for step in c['steps']:
   for needle in step.get('analysis_requires') or []:check_text(needle)
   if 'analysis_requires' in step and 'dynamic' not in step:
    for needle in step['analysis_requires']:
     if needle not in step['analysis']:raise ValueError(f'ungrounded authored analysis: {needle!r} in {c["id"]}')
 return cases,distribution

def freeze(args):
 if sha(args.manifest)!=MANIFEST_SHA:raise ValueError('tool authority')
 if args.records!=sum(count for _,count,_ in DISTRIBUTION):raise ValueError('records must equal the frozen v2 distribution total')
 cases,distribution=make_all_cases()
 minimum=args.minimum_verified_records if args.minimum_verified_records is not None else args.records
 if not 1<=minimum<=args.records:raise ValueError('minimum verified records')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
 plan={'schema':PLAN_SCHEMA,'seed':SEED,'records':args.records,'minimum_verified_records':minimum,'automatic_retry':False,
  'distribution':distribution,'system':SYSTEM,'tool_manifest':str(args.manifest.resolve()),'tool_manifest_sha256':sha(args.manifest),
  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  'authority_files':authority_files_v2(),'pi_bin':str(args.pi_bin.resolve()),
  'pi_version':subprocess.check_output([args.pi_bin,'--version'],text=True).strip(),
  'cases':cases,'model_generations':0,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False}
 publish(args.output/'plan-private.json',plan);print('HYBRID_V2_PLAN',args.records,minimum,sha(args.output/'plan-private.json'),flush=True)

# ------------------------------------------------------------------ execution
def resolve_dynamic_v2(spec,observation):
 """Resolve a dynamic step against the real observation; deterministic so the
 independent audit can re-derive the exact emitted frame from the recorded
 ToolResult."""
 if spec['dynamic']=='observation-exact':
  resolved=finish(observation.strip())
  resolved['analysis']=f'The tool observed {observation.strip()}; the user-facing answer is exactly that observed value.'
  resolved['commentary']=f'The observed answer is {observation.strip()}.'
  return resolved
 if spec['dynamic']=='observation-date':
  values=parse_date_observation(observation,spec['fmt'])
  message=spec['template'].format(**values)
  analysis=spec['analysis_template'].format(**values)
  if not message.strip() or not analysis.strip():raise ValueError('degenerate date finish')
  resolved=finish(message);resolved['analysis']=analysis;resolved['commentary']=spec['commentary']
  return resolved
 raise ValueError(f'unknown dynamic action {spec["dynamic"]}')

def execute_case_v2(case,panel,enc,root,pi_bin,manifest_path):
 workspace=root/'workspace';workspace.mkdir(parents=True)
 for name,text in case['files'].items():
  p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 fmt=case.get('date_fmt')
 utc=utc_family(fmt) if fmt else False
 clock_before=weekday_now(utc) if fmt else None
 index=0;emitted=[];emitted_specs=[]
 def generate(prompt,budget,deadline):
  nonlocal index
  spec=case['steps'][index]
  if 'dynamic' in spec:
   result=last_result(bridge)
   if result is None:raise ValueError('dynamic action without observation')
   spec=resolve_dynamic_v2(spec,result['content'][0]['text'])
  observed=last_result(bridge)['content'][0]['text'].strip() if last_result(bridge) else None
  comment=spec.get('commentary') if 'commentary' in spec else ('Using the verified observation.' if index and last_result(bridge) else None)
  analysis=spec.get('analysis')
  if analysis is not None:
   required=spec.get('analysis_requires') or []
   if not analysis.strip() or any(x not in analysis for x in required):raise ValueError('ungrounded analysis')
  text=frame(panel['tools'],spec['name'],spec['arguments'],comment,analysis);ids=enc.encode_ordinary(text)
  if len(ids)>budget:raise ValueError('authored turn budget')
  emitted.append(text);emitted_specs.append(spec);index+=1;return text,ids,'valid'
 bridge=NativePiToolBridge(panel,case['prompt'],enc,generate)
 terminal=serve_pi_native_tools(bridge,root/'pi',pi_bin=pi_bin,provider_extension=PROVIDER_V2,pi_extensions=EXTENSIONS,cwd=workspace,seconds=180,extra_env={'E97_PI_TOOL_MANIFEST':str(manifest_path.resolve())})
 if not terminal['close_verified'] or index!=len(case['steps']) or bridge.final is None:raise ValueError('incomplete authored trajectory')
 results=[m for m in bridge.history if m['role']=='toolResult'];calls=[b for m in bridge.history if m['role']=='assistant' for b in m['content'] if b['type']=='toolCall']
 expected_calls=[s for s in emitted_specs if s['name']!='finish']
 if len(results)!=len(calls) or any(r['toolCallId']!=c['id'] or r['toolName']!=c['name'] for c,r in zip(calls,results)):raise ValueError('call/result pairing')
 if len(calls)!=len(expected_calls) or any(c['name']!=s['name'] or c['arguments']!=s['arguments'] for c,s in zip(calls,expected_calls)):raise ValueError('Pi executed different action')
 if bridge.final!=emitted_specs[-1]['arguments']['message']:raise ValueError('terminal final mismatch')
 failed=lambda r:r['isError'] or r['content'][0]['text'].lower().startswith(('error:','tool error:')) or any(w in r['content'][0]['text'].lower() for w in ('exit code: 1','command exited with code 1','no matches found','no files found matching pattern'))
 if any(failed(r) for r in results):raise ValueError('unexpected tool error')
 snapshot={}
 for name,want in case['expected_files'].items():snapshot[name]=(workspace/name).read_text() if (workspace/name).exists() else None
 if snapshot!=case['expected_files']:raise ValueError('workspace oracle')
 if case.get('answer_must_be_observed'):
  if case.get('expected_answer') is None or not any(str(case['expected_answer']) in r['content'][0]['text'] for r in results):raise ValueError('answer not observed')
 if case.get('answer_is_observation'):
  if not results or bridge.final!=results[-1]['content'][0]['text'].strip():raise ValueError('answer is not the observation')
 clock_check=None
 if fmt:
  if not results:raise ValueError('date case without observation')
  observation=results[-1]['content'][0]['text']
  values=parse_date_observation(observation,fmt)
  clock_after=weekday_now(utc)
  if 'weekday' in values and values['weekday'] not in (clock_before,clock_after):raise ValueError(f'clock mismatch: observed {values["weekday"]} vs clock {clock_before}/{clock_after}')
  if 'ymd' in values and fmt=='u-ymd':
   span={clock_after}
   span.add((datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
   span.add((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
   if values['ymd'] not in span:raise ValueError(f'clock date mismatch: {values["ymd"]}')
  clock_check={'fmt':fmt,'weekday_before':clock_before,'weekday_after':clock_after,'observed':values,'final':bridge.final}
 ids,mask,units=encode_candidate(bridge.episode.text(),bridge.generations,case['supervise_from'],enc)
 private={'id':case['id'],'category':case['category'],'family':case['family'],'prompt':case['prompt'],'source_messages':bridge.episode.source_messages(),'native_record':bridge.episode.text(),'generations':bridge.generations,'public_history':bridge.history,'terminal':terminal,'snapshot':snapshot,'supervise_from':case['supervise_from'],'assistant_units':len(bridge.generations),'supervised_assistant_units':units,'targets':sum(mask),'record_sha256':hashlib.sha256(bridge.episode.text().encode()).hexdigest()}
 if clock_check is not None:private['clock_check']=clock_check
 publish(root/'episode-private.json',private);shutil.rmtree(workspace)
 return dict(id=case['id'],category=case['category'],family=case['family'],tokens=np.asarray(ids,dtype='<u4').tobytes(),mask=mask,targets=sum(mask),assistant_units=len(bridge.generations),supervised_units=units,errors=sum(r['isError'] for r in results),calls=len(calls),episode_sha256=sha(root/'episode-private.json'),sequence_sha256=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()+mask).hexdigest(),repository_discovery=case['repository_discovery'])

def row_from_private(case,private_path,enc):
 private=json.loads(private_path.read_text())
 ids,mask,units=encode_candidate(private['native_record'],private['generations'],private['supervise_from'],enc)
 if private['targets']!=sum(mask) or private['supervised_assistant_units']!=units:raise ValueError('resume re-encode mismatch')
 results=[m for m in private['public_history'] if m['role']=='toolResult']
 calls=[b for m in private['public_history'] if m['role']=='assistant' for b in m['content'] if b['type']=='toolCall']
 return dict(id=private['id'],category=private['category'],family=private['family'],tokens=np.asarray(ids,dtype='<u4').tobytes(),mask=mask,targets=sum(mask),assistant_units=private['assistant_units'],supervised_units=units,errors=sum(r['isError'] for r in results),calls=len(calls),episode_sha256=sha(private_path),sequence_sha256=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()+mask).hexdigest(),repository_discovery=False)

def collect(args):
 from scripts.e97_pi_native_bridge import BridgeStopped
 plan=json.loads(args.plan.read_text())
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 manifest_path=Path(plan['tool_manifest'])
 if sha(manifest_path)!=plan['tool_manifest_sha256']!=MANIFEST_SHA or plan['authority_files']!=authority_files_v2() or str(args.pi_bin.resolve())!=plan['pi_bin'] or subprocess.check_output([args.pi_bin,'--version'],text=True).strip()!=plan['pi_version'] or plan.get('automatic_retry',False) or plan['model_generations'] or plan['optimizer_updates']:raise ValueError('authority')
 if plan['schema']!=PLAN_SCHEMA:raise ValueError('plan schema')
 manifest=json.loads(manifest_path.read_text())
 panel={'system':plan['system'],'tools':manifest['model_visible_tools'],'max_turns':10,'generation_budget':2048,'episode_generation_budget':8192,'episode_seconds':150}
 args.output.mkdir(parents=True,mode=0o700,exist_ok=True)
 if (args.output/'candidate-authority/manifest.json').exists() and (args.output/'summary.json').exists():
  print('HYBRID_V2_ALREADY_COLLECTED',sha(args.output/'summary.json'));return
 enc=tiktoken.get_encoding('p50k_base')
 selected=plan['cases'] if args.stride==1 else plan['cases'][::args.stride]
 attempts=selected[:args.max_cases] if args.max_cases is not None else selected;total=len(attempts)
 resume_path=args.output/'resume-state.json'
 transient_counts={}
 if resume_path.exists():transient_counts=json.loads(resume_path.read_text()).get('transient_failures',{})
 stop=threading.Event()
 def attempt(case):
  root=args.output/case['id']
  if (root/'episode-private.json').exists():return row_from_private(case,root/'episode-private.json',enc),None
  if (root/'rejection.json').exists():return None,json.loads((root/'rejection.json').read_text())
  if stop.is_set():raise KeyboardInterrupt
  if root.exists():shutil.rmtree(root)
  try:
   return execute_case_v2(case,panel,enc,root,args.pi_bin,manifest_path),None
  except ValueError as exc:
   rejection={'id':case['id'],'category':case['category'],'family':case['family'],'type':'ValueError','message':str(exc),'retried':False}
   publish(root/'rejection.json',rejection);return None,rejection
  except (BridgeStopped,OSError,TimeoutError) as exc:
   # Transient transport/backend failure: keep the case resumable, never
   # silently drop it. Deterministic after two consecutive transient rounds.
   count=transient_counts.get(case['id'],0)+1
   transient_counts[case['id']]=count
   if root.exists():shutil.rmtree(root)
   if count>=2:
    rejection={'id':case['id'],'category':case['category'],'family':case['family'],'type':'BridgeStopped','message':f'transient transport failure x{count}: {exc}','retried':False}
    publish(root/'rejection.json',rejection);return None,rejection
   return None,{'id':case['id'],'transient':str(exc)}
 rows=[];rejections=[];done=0
 interrupted=None
 pool=ThreadPoolExecutor(max_workers=args.workers)
 futures=[pool.submit(attempt,case) for case in attempts]
 try:
  for future in futures:
   row,rejection=future.result();done+=1
   if row is not None:rows.append(row)
   elif rejection is not None and 'transient' not in rejection:rejections.append(rejection)
   if done%20==0 or done==total:
    print('HYBRID_V2_ATTEMPTS',done,'VERIFIED',len(rows),'REJECTED',len(rejections),'TRANSIENT',len(transient_counts),flush=True)
 except (KeyboardInterrupt,TimeoutError) as exc:
  interrupted=repr(exc);stop.set();print('HYBRID_V2_INTERRUPTED checkpointing',flush=True)
 finally:
  pool.shutdown(wait=True,cancel_futures=True)
 pending=[c['id'] for c in plan['cases'] if not ((args.output/c['id']/'episode-private.json').exists() or (args.output/c['id']/'rejection.json').exists())]
 resume_state={'schema':'emender-e97-hybrid-conversation-resume-v1','plan_sha256':args.plan_sha,'verified':len(rows),'rejected':len(rejections),'pending':pending,'transient_failures':transient_counts,'interrupted':interrupted}
 resume_path.write_text(json.dumps(resume_state,sort_keys=True,indent=1)+'\n')
 if interrupted is not None or pending:raise SystemExit(f'HYBRID_V2_INCOMPLETE verified={len(rows)} rejected={len(rejections)} pending={len(pending)}')
 (args.output/'rejections.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rejections))
 if len(rows)<plan['minimum_verified_records']:raise ValueError('minimum verified records not reached')
 if len({r['sequence_sha256'] for r in rows})!=len(rows):raise ValueError('dedup')
 write_authority(rows,{**plan,'plan_sha256':args.plan_sha},args.output)
 families=Counter(r['family'] for r in rows)
 summary={'schema':'emender-e97-hybrid-conversation-summary-v2','status':'qualified-candidates-not-admitted','attempted_records':plan['records'],'records':len(rows),'rejected_records':len(rejections),'automatic_retries':0,'minimum_verified_records':plan['minimum_verified_records'],'family_counts':dict(families),'repository_discovery_records':sum(r['repository_discovery'] for r in rows),'native_calls':sum(r['calls'] for r in rows),'authentic_tool_errors':sum(r['errors'] for r in rows),'assistant_targets':sum(r['targets'] for r in rows),'deduplicated_sequences':len(rows),'model_generations':0,'optimizer_updates':0,'training_eligible':False,'packing_authorized':False,'checkpoint_promotion':False,'authority_sha256':sha(args.output/'candidate-authority/manifest.json')}
 publish(args.output/'summary.json',summary)
 print('HYBRID_V2_COLLECTED',len(rows),len(rejections),summary['native_calls'],summary['assistant_targets'],flush=True)

def main():
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')))
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
 f=sub.add_parser('freeze');f.add_argument('--manifest',type=Path,required=True);f.add_argument('--records',type=int,required=True);f.add_argument('--minimum-verified-records',type=int);f.add_argument('--pi-bin',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
 c=sub.add_parser('collect');c.add_argument('--plan',type=Path,required=True);c.add_argument('--plan-sha',required=True);c.add_argument('--pi-bin',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--workers',type=int,default=1);c.add_argument('--max-cases',type=int,default=None,help='pilot bound: attempt only the first N selected cases (resume-friendly checkpoint; the run stays incomplete)');c.add_argument('--stride',type=int,default=1,help='pilot sampling: attempt every Nth plan case (1 = full plan)')
 a=parser.parse_args();freeze(a) if a.command=='freeze' else collect(a)
if __name__=='__main__':main()

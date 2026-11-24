from .k15_relation import reads
from pathlib import Path
from .k15_bookmarks import BookmarkSession
EXE=Path.home()/"sxgc/xsa/target/debug/xsa"


def get(raw,mode):
    session=BookmarkSession(EXE)
    try: return reads(raw,session,mode,True)[0]
    finally: session.close()


def test_saved_set_refines_after_new_byte_without_future_input():
    raw=b'xyABCxyAXY'+b'0'*20+b'xyABCZZ'; rows=get(raw,'full'); assert rows[0][:2]==(31,[65,65]); assert rows[1][:2]==(32,[66,88]); assert rows[2][:2]==(33,[67]); assert get(raw[:33],'full')==rows[:2]; assert get(raw,'immediate')==rows[:1]; assert get(raw,'scramble')!=rows; assert get(raw,'echo')[0][1]==[121,121]

# Clean Old Emender Hypervolume Previews

Task: `clean-old-emender`
Agent: `agent-588`
Date: 2026-05-30

## Scope Note

The literal remote directory `/home/erik/ndm` did not exist on Hypervolume.
The public old `ndm` URL, `http://hypervolu.me/~erik/ndm/`, maps to
`/home/erik/www/ndm`, which contained the old Emender paper PDFs. Cleanup was
limited to that public old `ndm` directory and filenames matching
`Garrison_2026_Emender*.pdf`.

No files were deleted. Old suffixed Emender candidate/preview PDFs were moved
aside into:

`/home/erik/www/ndm/.retired-emender-paper-previews-20260530T1835Z/`

The stable public PDF `/home/erik/www/ndm/Garrison_2026_Emender.pdf` was left
in place. Non-Emender `Garrison_2026_NDM*.pdf` and `Garrison_2026_PNR*.pdf`
files in `/home/erik/www/ndm` were not moved.

## New Candidate Verification

Remote path check:

```text
/home/erik/www/emender/Garrison_2026_Emender-67f15587.pdf	849485 bytes	2026-05-30 18:26:48.052603985 +0000
```

HTTP HEAD check:

```text
URL: http://hypervolu.me/~erik/emender/Garrison_2026_Emender-67f15587.pdf
HTTP: 200
Content-Type: application/pdf
Content-Length: 849485
```

## Pre-Cleanup Listing

Command:

```sh
ssh erik@hypervolu.me \
  'find "$HOME/www/ndm" -maxdepth 1 -type f -name "Garrison_2026_Emender*.pdf" -printf "%f\t%s bytes\t%TY-%Tm-%Td %TH:%TM:%TS %TZ\n" | sort'
```

Result:

```text
Garrison_2026_Emender-04f5699a.pdf	912666 bytes	2026-05-29 20:45:18.6515007720 UTC
Garrison_2026_Emender-0d5f4adf.pdf	916520 bytes	2026-05-26 22:53:26.3769191090 UTC
Garrison_2026_Emender-192e6b63.pdf	923994 bytes	2026-05-27 04:23:37.2916168130 UTC
Garrison_2026_Emender-2701fb59.pdf	925310 bytes	2026-05-27 12:01:03.8896074970 UTC
Garrison_2026_Emender-277231c7.pdf	913384 bytes	2026-05-26 18:42:53.2159338910 UTC
Garrison_2026_Emender-2faa7648.pdf	919310 bytes	2026-05-27 20:33:13.9067209370 UTC
Garrison_2026_Emender-35410f8f.pdf	912386 bytes	2026-05-26 21:11:40.6369804080 UTC
Garrison_2026_Emender-3a062007.pdf	918960 bytes	2026-05-27 04:10:13.6999713300 UTC
Garrison_2026_Emender-3bd16a16.pdf	918066 bytes	2026-05-27 02:44:16.5824453160 UTC
Garrison_2026_Emender-450844be.pdf	851381 bytes	2026-05-29 23:13:59.5918106830 UTC
Garrison_2026_Emender-4a180e77.pdf	881691 bytes	2026-05-26 13:07:47.7041230390 UTC
Garrison_2026_Emender-4f790077.pdf	916870 bytes	2026-05-26 19:07:53.0190388950 UTC
Garrison_2026_Emender-50b5cd8f.pdf	924484 bytes	2026-05-27 15:26:22.3906307960 UTC
Garrison_2026_Emender-56511004.pdf	883602 bytes	2026-05-26 00:18:00.5326066950 UTC
Garrison_2026_Emender-5ba01369.pdf	883602 bytes	2026-05-26 00:41:42.6078706480 UTC
Garrison_2026_Emender-626c5e8a.pdf	931123 bytes	2026-05-28 14:09:09.2841795480 UTC
Garrison_2026_Emender-6983d88e.pdf	916774 bytes	2026-05-26 19:05:01.6135449340 UTC
Garrison_2026_Emender-6c03136e.pdf	917222 bytes	2026-05-27 00:48:15.5730591670 UTC
Garrison_2026_Emender-6c9c5bc4.pdf	912766 bytes	2026-05-29 18:30:03.5629052620 UTC
Garrison_2026_Emender-6d83d376.pdf	930461 bytes	2026-05-28 13:09:04.5691597200 UTC
Garrison_2026_Emender-74f24bce.pdf	918892 bytes	2026-05-27 04:00:49.2070336810 UTC
Garrison_2026_Emender-7b407726.pdf	883611 bytes	2026-05-25 20:17:42.5318658720 UTC
Garrison_2026_Emender-7c922f7c-dirty.pdf	755721 bytes	2026-05-25 14:46:50.1650381240 UTC
Garrison_2026_Emender-83a34698.pdf	930268 bytes	2026-05-28 13:36:58.0280204840 UTC
Garrison_2026_Emender-8c0352b9.pdf	912766 bytes	2026-05-29 18:28:55.7888444010 UTC
Garrison_2026_Emender-8f4830ce.pdf	853401 bytes	2026-05-25 16:31:35.4166098720 UTC
Garrison_2026_Emender-8fd6d1de.pdf	914180 bytes	2026-05-28 05:22:34.6490795560 UTC
Garrison_2026_Emender-9acdfa76.pdf	914121 bytes	2026-05-28 06:00:04.7199243590 UTC
Garrison_2026_Emender-9bdfc4da.pdf	925310 bytes	2026-05-27 04:46:24.2688854460 UTC
Garrison_2026_Emender-a5454129.pdf	900465 bytes	2026-05-26 17:43:13.6621665940 UTC
Garrison_2026_Emender-b0feac7b.pdf	930904 bytes	2026-05-28 12:12:22.0000000000 UTC
Garrison_2026_Emender-b9172dd9.pdf	912386 bytes	2026-05-26 19:18:06.6076308770 UTC
Garrison_2026_Emender-c240e8fa.pdf	924413 bytes	2026-05-27 13:51:07.3160859360 UTC
Garrison_2026_Emender-cbc5fff0.pdf	924481 bytes	2026-05-27 15:09:02.7723142420 UTC
Garrison_2026_Emender-cc03096d.pdf	935311 bytes	2026-05-25 23:12:47.5947756330 UTC
Garrison_2026_Emender-ce26f193.pdf	883611 bytes	2026-05-25 23:47:32.9110793810 UTC
Garrison_2026_Emender-d5c6aa6f.pdf	924884 bytes	2026-05-27 04:34:24.2331686670 UTC
Garrison_2026_Emender-d7354913.pdf	918066 bytes	2026-05-27 02:03:31.2664088880 UTC
Garrison_2026_Emender-d83f64b4.pdf	900621 bytes	2026-05-26 13:42:19.4653080980 UTC
Garrison_2026_Emender-e0196909.pdf	921496 bytes	2026-05-28 06:51:09.4719940190 UTC
Garrison_2026_Emender-eef0b357.pdf	878874 bytes	2026-05-25 19:17:41.1359819410 UTC
Garrison_2026_Emender-fd97cc59.pdf	912666 bytes	2026-05-29 20:48:27.9473762680 UTC
Garrison_2026_Emender-fdd3a713.pdf	851922 bytes	2026-05-25 16:28:44.9106089430 UTC
Garrison_2026_Emender-figure-label-gdn2-preview-20260530T162352Z.pdf	852189 bytes	2026-05-30 16:24:01.2567536860 UTC
Garrison_2026_Emender-figure2-legend-cleanup-preview-20260530T174010Z.pdf	849206 bytes	2026-05-30 17:40:08.9718495620 UTC
Garrison_2026_Emender-label-nudge-preview.pdf	852141 bytes	2026-05-29 23:31:45.9928067800 UTC
Garrison_2026_Emender.pdf	912666 bytes	2026-05-29 20:48:27.9703766200 UTC
```

## Files Moved Aside

The following old suffixed Emender PDFs were moved from
`/home/erik/www/ndm/` into
`/home/erik/www/ndm/.retired-emender-paper-previews-20260530T1835Z/`:

```text
Garrison_2026_Emender-04f5699a.pdf
Garrison_2026_Emender-0d5f4adf.pdf
Garrison_2026_Emender-192e6b63.pdf
Garrison_2026_Emender-2701fb59.pdf
Garrison_2026_Emender-277231c7.pdf
Garrison_2026_Emender-2faa7648.pdf
Garrison_2026_Emender-35410f8f.pdf
Garrison_2026_Emender-3a062007.pdf
Garrison_2026_Emender-3bd16a16.pdf
Garrison_2026_Emender-450844be.pdf
Garrison_2026_Emender-4a180e77.pdf
Garrison_2026_Emender-4f790077.pdf
Garrison_2026_Emender-50b5cd8f.pdf
Garrison_2026_Emender-56511004.pdf
Garrison_2026_Emender-5ba01369.pdf
Garrison_2026_Emender-626c5e8a.pdf
Garrison_2026_Emender-6983d88e.pdf
Garrison_2026_Emender-6c03136e.pdf
Garrison_2026_Emender-6c9c5bc4.pdf
Garrison_2026_Emender-6d83d376.pdf
Garrison_2026_Emender-74f24bce.pdf
Garrison_2026_Emender-7b407726.pdf
Garrison_2026_Emender-7c922f7c-dirty.pdf
Garrison_2026_Emender-83a34698.pdf
Garrison_2026_Emender-8c0352b9.pdf
Garrison_2026_Emender-8f4830ce.pdf
Garrison_2026_Emender-8fd6d1de.pdf
Garrison_2026_Emender-9acdfa76.pdf
Garrison_2026_Emender-9bdfc4da.pdf
Garrison_2026_Emender-a5454129.pdf
Garrison_2026_Emender-b0feac7b.pdf
Garrison_2026_Emender-b9172dd9.pdf
Garrison_2026_Emender-c240e8fa.pdf
Garrison_2026_Emender-cbc5fff0.pdf
Garrison_2026_Emender-cc03096d.pdf
Garrison_2026_Emender-ce26f193.pdf
Garrison_2026_Emender-d5c6aa6f.pdf
Garrison_2026_Emender-d7354913.pdf
Garrison_2026_Emender-d83f64b4.pdf
Garrison_2026_Emender-e0196909.pdf
Garrison_2026_Emender-eef0b357.pdf
Garrison_2026_Emender-fd97cc59.pdf
Garrison_2026_Emender-fdd3a713.pdf
Garrison_2026_Emender-figure-label-gdn2-preview-20260530T162352Z.pdf
Garrison_2026_Emender-figure2-legend-cleanup-preview-20260530T174010Z.pdf
Garrison_2026_Emender-label-nudge-preview.pdf
```

## Post-Cleanup Listing

Command:

```sh
ssh erik@hypervolu.me \
  'find "$HOME/www/ndm" -maxdepth 1 -type f -name "Garrison_2026_Emender*.pdf" -printf "%f\t%s bytes\t%TY-%Tm-%Td %TH:%TM:%TS %TZ\n" | sort'
```

Result:

```text
Garrison_2026_Emender.pdf	912666 bytes	2026-05-29 20:48:27.9703766200 UTC
```

Stable PDF HTTP check:

```text
URL: http://hypervolu.me/~erik/ndm/Garrison_2026_Emender.pdf
HTTP: 200
Content-Type: application/pdf
```

Representative retired preview old URL check:

```text
URL: http://hypervolu.me/~erik/ndm/Garrison_2026_Emender-figure-label-gdn2-preview-20260530T162352Z.pdf
HTTP: 404
```

Public directory listing check:

```text
curl http://hypervolu.me/~erik/ndm/ | grep -Eo 'Garrison_2026_Emender[^"<]*\.pdf|retired-emender[^"<]*' | sort
Garrison_2026_Emender.pdf
Garrison_2026_Emender.pdf
```

The duplicate `Garrison_2026_Emender.pdf` entry is the link text plus href from
the Apache directory index. No retired directory or suffixed Emender PDF appeared
in the public index.

## Validation Summary

- Recorded the pre-cleanup listing of matching old Emender PDFs in the old
  public `ndm` directory.
- Verified the new git-versioned candidate exists under
  `/home/erik/www/emender/` and returns HTTP 200 as `application/pdf`.
- Moved aside the old suffixed `Garrison_2026_Emender-*.pdf` files into the
  hidden retired directory above.
- Recorded the post-cleanup listing: only stable
  `Garrison_2026_Emender.pdf` remains at top level.
- Verified stable `Garrison_2026_Emender.pdf` remains available at HTTP 200.
- Left unrelated `Garrison_2026_NDM*.pdf` and `Garrison_2026_PNR*.pdf` files in
  place in the old public `ndm` directory.

No cargo or pytest run was needed because this task changed only remote paper
artifacts plus this provenance record; no project code was modified.

[app]
title = DocTranslator
package.name = doctranslator
package.domain = org.test
source.dir = .
source.include_exts = py,json
version = 2.0

# Pinned versions for stability in the build environment
requirements = python3,kivy,lxml==4.9.3,requests,torch==1.13.1,transformers,sentencepiece,python-docx,pyjnius==1.6.1,Cython==0.29.36,Pillow

orientation = portrait
fullscreen = 0

[android]
android.permissions = READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, INTERNET
android.api = 31
android.minapi = 24
# The GitHub Actions runner has the correct SDK/NDK versions, so we don't need to specify them.
# android.sdk = 24
# android.ndk = 23c
android.arch = arm64-v8a

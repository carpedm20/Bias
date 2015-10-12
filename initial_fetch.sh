#!/bin/bash
wget -U 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.1.6) Gecko/20070802 SeaMonkey/1.1.4' http://cdn.mustar.kr/namuwiki150928.7z -O data/wiki.7z
7z x data/wiki.7z -o data/

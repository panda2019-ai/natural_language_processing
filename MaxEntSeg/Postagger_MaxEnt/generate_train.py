#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys

for line in sys.stdin:
	rec = line.strip().split()

	for i in range(len(rec)):
		wo_tag = rec[i]
		wo_tag_rec = wo_tag.split("/")
		wo = wo_tag_rec[0]
		tag = wo_tag_rec[1]
		word = word_b = word_e = wo

		if i-1 >= 0:
			pre1 = pre1_b = pre1_e = (rec[i-1].split("/"))[0]
		else:
			pre1 = pre1_b = pre1_e = "#"

		if i-2 >= 0:
			pre2 = pre2_b = pre2_e = (rec[i-2].split("/"))[0]
		else:
			pre2 = pre2_b = pre2_e = "#"

		if i+1 < len(rec):
			aft1 = aft1_b = aft1_e = (rec[i+1].split("/"))[0]
		else:
			aft1 = aft1_b = aft1_e = "#"

		if i+2 < len(rec):
			aft2 = aft2_b = aft2_e = (rec[i+2].split("/"))[0]
		else:
			aft2 = aft2_b = aft2_e = "#"

		result = tag + "\tword=" + word + "\tword_b=" + word_b + "\tword_e=" + word_e + "\tpre1=" + pre1 + "\tpre1_b=" + pre1_b + "\tpre1_e=" + pre1_e + "\tpre2=" + pre2 + "\tpre2_b=" + pre2_b + "\tpre2_e=" + pre2_e + "\taft1=" + aft1 + "\taft1_b=" + aft1_b + "\taft1_e=" + aft1_e + "\taft2=" + aft2 + "\taft2_b=" + aft2_b + "\taft2_e=" + aft2_e
		print result

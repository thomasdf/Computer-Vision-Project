import hashlib

import sys

_hash = lambda text, start, end, length: int(hashlib.sha1(text[start:end].encode('utf-8')).hexdigest(), 16) % length

al_fathia_english = 'Praise be to Allah, Lord of the Worlds, (2) The Beneficent, the Merciful. (3) Owner of the Day of Judgment, (4) Thee (alone) we worship; Thee (alone) we ask for help. (5) Show us the straight path, (6) The path of those whom Thou hast favoured. Not (the path) of those who earn Thine anger nor of those who go astray. (7)'

al_fathia_arabic = 'ٱلۡحَمۡدُ لِلَّهِ رَبِّ ٱلۡعَـٰلَمِينَ (﻿٢﻿) ٱلرَّحۡمَـٰنِ ٱلرَّحِيمِ (﻿٣﻿) مَـٰلِكِ يَوۡمِ ٱلدِّينِ (﻿٤﻿) إِيَّاكَ نَعۡبُدُ وَإِيَّاكَ نَسۡتَعِينُ (﻿٥﻿) ٱهۡدِنَا ٱلصِّرَٲطَ ٱلۡمُسۡتَقِيمَ (﻿٦﻿) صِرَٲطَ ٱلَّذِينَ أَنۡعَمۡتَ عَلَيۡهِمۡ غَيۡرِ ٱلۡمَغۡضُوبِ عَلَيۡهِمۡ وَلَا ٱلضَّآلِّينَ (﻿٧﻿)'

vatsyayana_bio = '''Mallanaga Vatsyayana was a very holy man (sadhu), a seer, and a sage (rishi), and in all of the spiritual senses of the word, a tantric. Mallanaga worshipped the Divine as both feminine and masculine (Shaktishiva), and lived primarily a religious life. Mallanaga wrote the Kama Sutra for the ruling class (nobled rulers, lords, princes and kings), which at that time in India's history was the Kshatriya, or Warrior caste. Based on mentions of 1st Century historical figures in the Kama Sutra, and on mentions of the Kama Sutra in early 5th Century works, we know that Mallanaga Vatsyayana wrote the Sutra sometime between the 1st and 4th Centuries A.D. The Kama Sutra is simultaneously a manual of matchmaking, flirting, sensuality in life and in sex, romantic love, human nature, attracting a man, turning on a woman, how to seduce a man, how to captivate a woman, how to get a man or woman to marry you, arranged marriages, affairs, gold-digging, the economics of love, affairs with courtesans, keeping the affections of a lover or spouse, love potions, charms, and everything in between. Mallanaga Vatsyayana not include deeper tantric sexual practices in his most famous work, because he knew that sexuality is only an appropriate spiritual tool for some good students of tantra marga. Mallanaga wrote the Kama Sutra for the ruling class and their educare so they could balance and enjoy their sensual appetites with their social and spiritual obligations as rulers. And He as a seer not to pass on secrets he knew would be lost on many of these students'''


scripture = (vatsyayana_bio, al_fathia_arabic, al_fathia_english)

def hash_split(test_part: float, set_size: int, result_size: int = -1, hash = _hash):
	if not(0.0 < test_part < 1.0):
		raise Exception('Dafaq m8?')

	if result_size == -1:
		result_size = set_size

	result = []
	test_size = int(result_size * test_part)
	num = 0
	q = scripture[num]
	scripture_len = len(scripture)
	while len(q) < test_size:
		num += 1
		q += scripture[num % scripture_len]
	num = 0
	for i in range(test_size):

		p = hash(q, 0, i, set_size)

		while p in result:
			p = hash(q, num, i, set_size)
			num += 1
		num = 0
		result.append(p)

	return result

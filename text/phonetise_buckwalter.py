import re

arabic_to_buckw_dict = {  # mapping from Arabic script to Buckwalter
    u'\u0628': u'b', u'\u0630': u'*', u'\u0637': u'T', u'\u0645': u'm',
    u'\u062a': u't', u'\u0631': u'r', u'\u0638': u'Z', u'\u0646': u'n',
    u'\u062b': u'^', u'\u0632': u'z', u'\u0639': u'E', u'\u0647': u'h',
    u'\u062c': u'j', u'\u0633': u's', u'\u063a': u'g', u'\u062d': u'H',
    u'\u0642': u'q', u'\u0641': u'f', u'\u062e': u'x', u'\u0635': u'S',
    u'\u0634': u'$', u'\u062f': u'd', u'\u0636': u'D', u'\u0643': u'k',
    u'\u0623': u'>', u'\u0621': u'\'', u'\u0626': u'}', u'\u0624': u'&',
    u'\u0625': u'<', u'\u0622': u'|', u'\u0627': u'A', u'\u0649': u'Y',
    u'\u0629': u'p', u'\u064a': u'y', u'\u0644': u'l', u'\u0648': u'w',
    u'\u064b': u'F', u'\u064c': u'N', u'\u064d': u'K', u'\u064e': u'a',
    u'\u064f': u'u', u'\u0650': u'i', u'\u0651': u'~', u'\u0652': u'o'
}

buckw_to_arabic_dict = {  # mapping from Buckwalter to Arabic script
    u'b': u'\u0628', u'*': u'\u0630', u'T': u'\u0637', u'm': u'\u0645',
    u't': u'\u062a', u'r': u'\u0631', u'Z': u'\u0638', u'n': u'\u0646',
    u'^': u'\u062b', u'z': u'\u0632', u'E': u'\u0639', u'h': u'\u0647',
    u'j': u'\u062c', u's': u'\u0633', u'g': u'\u063a', u'H': u'\u062d',
    u'q': u'\u0642', u'f': u'\u0641', u'x': u'\u062e', u'S': u'\u0635',
    u'$': u'\u0634', u'd': u'\u062f', u'D': u'\u0636', u'k': u'\u0643',
    u'>': u'\u0623', u'\'': u'\u0621', u'}': u'\u0626', u'&': u'\u0624',
    u'<': u'\u0625', u'|': u'\u0622', u'A': u'\u0627', u'Y': u'\u0649',
    u'p': u'\u0629', u'y': u'\u064a', u'l': u'\u0644', u'w': u'\u0648',
    u'F': u'\u064b', u'N': u'\u064c', u'K': u'\u064d', u'a': u'\u064e',
    u'u': u'\u064f', u'i': u'\u0650', u'~': u'\u0651', u'o': u'\u0652'
}

def arabic_to_buckwalter(word):
    res = ''
    for letter in word:
        if letter in arabic_to_buckw_dict:
            res += arabic_to_buckw_dict[letter]
        else:
            res += letter
    return res

def buckwalter_to_arabic(word):
    res = ''
    for letter in word:
        if letter in buckw_to_arabic_dict:
            res += buckw_to_arabic_dict[letter]
        else:
            res += letter
    return res

# ----------------------------------------------------------------------------
# Grapheme to Phoneme mappings
# ----------------------------------------------------------------------------
unambiguousConsonantMap = {
    u'b': u'b', u'*': u'*', u'T': u'T', u'm': u'm',
    u't': u't', u'r': u'r', u'Z': u'Z', u'n': u'n',
    u'^': u'^', u'z': u'z', u'E': u'E', u'h': u'h',
    u'j': u'j', u's': u's', u'g': u'g', u'H': u'H',
    u'q': u'q', u'f': u'f', u'x': u'x', u'S': u'S',
    u'$': u'$', u'd': u'd', u'D': u'D', u'k': u'k',
    u'>': u'<', u'\'': u'<', u'}': u'<', u'&': u'<',
    u'<': u'<'
}

ambiguousConsonantMap = {
    u'l': [u'l', u''], u'w': u'w', u'y': u'y', u'p': [u't', u'']
}

maddaMap = {
    u'|': [[u'<', u'aa'], [u'<', u'AA']]
}

# Mise à jour pour inclure toutes les voyelles de symbols.py
vowelMap = {
    u'A': [[u'aa', u''], [u'AA', u'']],
    u'Y': [[u'aa', u''], [u'AA', u'']],
    u'w': [[u'uu', u'uu0'], [u'UU0', u'UU1']],
    u'y': [[u'ii', u'ii0'], [u'II0', u'II1']],
    u'a': [u'a', u'A'],
    u'u': [[u'u', u'u0'], [u'u1', u'U0']],
    u'i': [[u'i', u'i0'], [u'i1', u'I0']],
}

nunationMap = {
    u'F': [[u'a', u'n'], [u'A', u'n']],
    u'N': [[u'u1', u'n'], [u'U1', u'n']],
    u'K': [[u'i1', u'n'], [u'I1', u'n']]
}

diacritics = [u'o', u'a', u'u', u'i', u'F', u'N', u'K', u'~']
diacriticsWithoutShadda = [u'o', u'a', u'u', u'i', u'F', u'N', u'K']
emphatics = [u'D', u'S', u'T', u'Z', u'g', u'x', u'q']
forwardEmphatics = [u'g', u'x']
consonants = [u'>', u'<', u'}', u'&', u'\'', u'b', u't', u'^', u'j', u'H', u'x', u'd', u'*', u'r',
              u'z', u's', u'$', u'S', u'D', u'T', u'Z', u'E', u'g', u'f', u'q', u'k', u'l', u'm', u'n', u'h', u'|']

punctuation = ['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '-', '"', "'"]

# ------------------------------------------------------------------------------------
# Words with fixed irregular pronunciations
# ------------------------------------------------------------------------------------
fixedWords = {
    u'h*A': [u'h aa * aa', u'h aa * a'],
    u'h*h': [u'h aa * i0 h i0', u'h aa * i1 h'],
    u'h*An': [u'h aa * aa n i0', u'h aa * aa n'],
    u'h&lA\'': [u'h aa < u0 l aa < i0', u'h aa < u0 l aa <'],
    u'*lk': [u'* aa l i0 k a', u'* aa l i0 k'],
    u'k*lk': [u'k a * aa l i0 k a', u'k a * aa l i1 k'],
    u'*lkm': u'* aa l i0 k u1 m',
    u'>wl}k': [u'< u0 l aa < i0 k a', u'< u0 l aa < i1 k'],
    u'Th': u'T aa h a',
    u'lkn': [u'l aa k i0 n _dbl_ n a', u'l aa k i1 n'],  # Utilisation de _dbl_
    u'lknh': u'l aa k i0 n _dbl_ n a h u0',
    u'lknhm': u'l aa k i0 n _dbl_ n a h u1 m',
    u'lknk': [u'l aa k i0 n _dbl_ n a k a', u'l aa k i0 n _dbl_ n a k i0'],
    u'lknkm': u'l aa k i0 n _dbl_ n a k u1 m',
    u'lknkmA': u'l aa k i0 n _dbl_ n a k u0 m aa',
    u'lknnA': u'l aa k i0 n _dbl_ n a n aa',
    u'AlrHmn': [u'r _dbl_ r a H m aa n i0',  u'r _dbl_ r a H m aa n'],
    u'Allh': [u'l _dbl_ l aa h i0', u'l _dbl_ l aa h', u'l _dbl_ l AA h u0', u'l _dbl_ l AA h a', u'l _dbl_ l AA h', u'l _dbl_ l A'],
    u'h*yn': [u'h aa * a y n i0', u'h aa * a y n'],
    u'nt': u'n i1 t',
    u'fydyw': u'v i0 d y uu1',
    u'lndn': u'l A n d u1 n'
}

def isFixedWord(word, results, orthography, pronunciations):
    lastLetter = ''
    if len(word) > 0:
        lastLetter = word[-1]
    if lastLetter == u'a':
        lastLetter = [u'a', u'A']
    elif lastLetter == u'A':
        lastLetter = [u'aa']
    elif lastLetter == u'u':
        lastLetter = [u'u0']
    elif lastLetter == u'i':
        lastLetter = [u'i0']
    elif lastLetter in unambiguousConsonantMap:
        lastLetter = [unambiguousConsonantMap[lastLetter]]

    wordConsonants = re.sub(u'[^h*Ahn\'>wl}kmyTtfd]', '', word)
    if wordConsonants in fixedWords:
        if isinstance(fixedWords[wordConsonants], list):
            for pronunciation in fixedWords[wordConsonants]:
                if pronunciation.split(' ')[-1] in lastLetter:
                    results += word + ' ' + pronunciation + '\n'
                    pronunciations.append(pronunciation.split(' '))
        else:
            results += word + ' ' + fixedWords[wordConsonants] + '\n'
            pronunciations.append(fixedWords[wordConsonants].split(' '))
    return results

def preprocess_utterance(utterance):
    utterance = utterance.replace(u'AF', u'F')
    utterance = utterance.replace(u'\u0640', u'')
    utterance = utterance.replace(u'o', u'')
    utterance = utterance.replace(u'aA', u'A')
    utterance = utterance.replace(u'aY', u'Y')
    utterance = utterance.replace(u' A', u' ')
    utterance = utterance.replace(u'F', u'an')
    utterance = utterance.replace(u'N', u'un')
    utterance = utterance.replace(u'K', u'in')
    utterance = utterance.replace(u'|', u'>A')

    utterance = utterance.replace('i~', '~i')
    utterance = utterance.replace('a~', '~a')
    utterance = utterance.replace('u~', '~u')

    utterance = re.sub(u'Ai', u'<i', utterance)
    utterance = re.sub(u'Aa', u'>a', utterance)
    utterance = re.sub(u'Au', u'>u', utterance)
    utterance = re.sub(u'^>([^auAw])', u'>a\\1', utterance)
    utterance = re.sub(u' >([^auAw ])', u' >a\\1', utterance)
    utterance = re.sub(u'<([^i])', u'<i\\1', utterance)

    utterance = re.sub(r"(\S)([.,;:!?\(\)\[\]\{\}\-\"'])", r"\1 \2", utterance)

    utterance = utterance.split(u' ')
    return utterance

def process_word(word):
    if word in punctuation:
        return word

    pronunciations = []
    isFixedWord(word, '', word, pronunciations)

    emphaticContext = False
    word = u'bb' + word + u'ee'
    phones = []

    for index in range(2, len(word) - 2):
        letter = word[index]
        letter1 = word[index + 1]
        letter2 = word[index + 2]
        letter_1 = word[index - 1]
        letter_2 = word[index - 2]

        if letter in consonants + [u'w', u'y'] and letter not in emphatics + [u'r', u'l']:
            emphaticContext = False
        if letter in emphatics:
            emphaticContext = True
        if letter1 in emphatics and letter1 not in forwardEmphatics:
            emphaticContext = True

        if letter in unambiguousConsonantMap:
            phones += [unambiguousConsonantMap[letter]]
        if letter == u'l':
            if (letter1 not in diacritics and letter1 not in vowelMap) and letter2 in [u'~']:
                phones += [ambiguousConsonantMap[u'l'][1]]
            else:
                phones += [ambiguousConsonantMap[u'l'][0]]
        if letter == u'~' and letter_1 not in [u'w', u'y'] and len(phones) > 0:
            phones[-1] = phones[-1] + ' _dbl_ ' + phones[-1]  # Utilisation de _dbl_
        if letter == u'|':
            if emphaticContext:
                phones += maddaMap[u'|'][1]
            else:
                phones += maddaMap[u'|'][0]
        if letter == u'p':
            if letter1 in diacritics:
                phones += [ambiguousConsonantMap[u'p'][0]]
            else:
                phones += [ambiguousConsonantMap[u'p'][1]]
        if letter in vowelMap:
            if letter in [u'w', u'y']:
                if letter1 in diacriticsWithoutShadda + [u'A', u'Y'] or (letter1 in [u'w', u'y'] and letter2 not in diacritics + [u'A', u'w', u'y']) or (letter_1 in diacriticsWithoutShadda and letter1 in consonants + [u'e']):
                    if (letter == u'w' and letter_1 in [u'u'] and letter1 not in [u'a', u'i', u'A', u'Y']) or (letter == u'y' and letter_1 in [u'i'] and letter1 not in [u'a', u'u', u'A', u'Y']):
                        if emphaticContext:
                            phones += [vowelMap[letter][1][0]]
                        else:
                            phones += [vowelMap[letter][0][0]]
                    else:
                        if letter1 in [u'A'] and letter == u'w' and letter2 in [u'e']:
                            phones += [[ambiguousConsonantMap[letter], vowelMap[letter][0][0]]]
                        else:
                            phones += [ambiguousConsonantMap[letter]]
                elif letter1 in [u'~']:
                    if letter_1 in [u'a'] or (letter == u'w' and letter_1 in [u'i', u'y']) or (letter == u'y' and letter_1 in [u'w', u'u']):
                        phones += [ambiguousConsonantMap[letter], ambiguousConsonantMap[letter]]
                    else:
                        phones += [vowelMap[letter][0][0], ambiguousConsonantMap[letter]]
                else:
                    if emphaticContext:
                        if letter_1 in consonants + [u'u', u'i'] and letter1 in [u'e']:
                            phones += [[vowelMap[letter][1][0], vowelMap[letter][1][0][1:]]]
                        else:
                            phones += [vowelMap[letter][1][0]]
                    else:
                        if letter_1 in consonants + [u'u', u'i'] and letter1 in [u'e']:
                            phones += [[vowelMap[letter][0][0], vowelMap[letter][0][0][1:]]]
                        else:
                            phones += [vowelMap[letter][0][0]]
            if letter in [u'u', u'i']:
                if emphaticContext:
                    if (letter1 in unambiguousConsonantMap or letter1 == u'l') and letter2 == u'e' and len(word) > 7:
                        phones += [vowelMap[letter][1][1]]
                    else:
                        phones += [vowelMap[letter][1][0]]
                else:
                    if (letter1 in unambiguousConsonantMap or letter1 == u'l') and letter2 == u'e' and len(word) > 7:
                        phones += [vowelMap[letter][0][1]]
                    else:
                        phones += [vowelMap[letter][0][0]]
            if letter in [u'a', u'A', u'Y']:
                if letter == u'A' and letter_1 in [u'w', u'k'] and letter_2 == u'b':
                    phones += [[u'a', vowelMap[letter][0][0]]]
                elif letter == u'A' and letter_1 in [u'u', u'i']:
                    pass
                elif letter == u'A' and letter_1 in [u'w'] and letter1 in [u'e']:
                    phones += [[vowelMap[letter][0][0], vowelMap[letter][0][1]]]
                elif letter in [u'A', u'Y'] and letter1 in [u'e']:
                    if emphaticContext:
                        phones += [[vowelMap[letter][1][0], vowelMap[u'a'][1]]]
                    else:
                        phones += [[vowelMap[letter][0][0], vowelMap[u'a'][0]]]
                else:
                    if emphaticContext:
                        phones += [vowelMap[letter][1][0]]
                    else:
                        phones += [vowelMap[letter][0][0]]

    possibilities = 1
    for letter in phones:
        if isinstance(letter, list):
            possibilities *= len(letter)

    for i in range(possibilities):
        pronunciations.append([])
        iterations = 1
        for index, letter in enumerate(phones):
            if isinstance(letter, list):
                curIndex = int(i / iterations) % len(letter)
                if letter[curIndex] != u'':
                    pronunciations[-1].append(letter[curIndex])
                iterations *= len(letter)
            else:
                if letter != u'':
                    pronunciations[-1].append(letter)

    for pronunciation in pronunciations:
        prevLetter = ''
        toDelete = []
        for i in range(len(pronunciation)):
            letter = pronunciation[i]
            if letter in ['aa', 'uu', 'ii', 'AA', 'UU0', 'II0'] and prevLetter.lower() == letter[1:].lower():
                toDelete.append(i - 1)
                pronunciation[i] = pronunciation[i - 1][0] + pronunciation[i - 1]
            if letter in ['u', 'i'] and prevLetter.lower() == letter.lower():
                toDelete.append(i - 1)
                pronunciation[i] = pronunciation[i - 1]
            if letter in ['y', 'w'] and prevLetter == letter:
                pronunciation[i - 1] = pronunciation[i - 1] + ' _dbl_ ' + pronunciation[i - 1]
                toDelete.append(i)
            prevLetter = letter
        for i in reversed(range(len(toDelete))):
            del pronunciation[toDelete[i]]

    return pronunciations[0] if pronunciations else []

def process_utterance(utterance):
    utterance = preprocess_utterance(utterance)
    phonemes = []

    for word in utterance:
        if word in punctuation:
            phonemes.append([word])
        elif word in ['-', 'sil']:
            phonemes.append(['_sil_'])
        else:
            phonemes_word = process_word(word)
            phonemes.append(phonemes_word)

    final_sequence = ' _+_ '.join(' '.join(phon for phon in phones) for phones in phonemes)
    return final_sequence
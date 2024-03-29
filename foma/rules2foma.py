import re
import csv


LEXICON_FILE = 'dict.txt'
DEFINITIONS_FOR_RULES_FILE = 'definitions.csv'
RULES_FILE = 'rules.csv'
MORPH_FILE = 'morph.csv'

GUESS_SYMB = 'GUESS+'
pos_guess_symb_pat = r"\^[A-Z]+"


def read_csv_dict(file):
    rows = []
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def add_space(s):
    NOSPACE_CHARS = (' ', '[', ']', '"')

    res = ''
    need_space = True
    for ch in s:
        if ch == '"' and need_space:
            need_space = False
        elif ch == '"' and not need_space:
            need_space = True
        if re.match("[A-Za-z]", ch):
            res += ch
            continue
        res += f"""{ch}{' ' if need_space and ch not in NOSPACE_CHARS else ''}"""

    return res


def convert_definitions():
    res = ""
    rows = read_csv_dict(DEFINITIONS_FOR_RULES_FILE)

    for row in rows:
        res += (f"define {row['union']} "
                f"[ {'| '.join(add_space(val) for val in row['values'].split())}];\n")

    return res


def make_multichars_string(multichars):
    return f"Multichar_Symbols {' '.join(multichars)}"


def parse_pluses(s):
    res = []
    while '+' in s:
        s = s[s.find('+')+1:]
        cur_part = '+'

        for i, ch in enumerate(s):
            if ch in ('+', ' ', ':', '"'):
                break
            cur_part += ch

        res.append(cur_part)
        s = s[i:]
    return res


def convert_morph_get_multichars():
    res = ""
    lexicons_res = []
    multichars = []
    rows = read_csv_dict(MORPH_FILE)

    lexicons = {}
    for row in rows:
        lexicons.setdefault(row['current_class'], []).append(row)

    for lexicon, morph_list in lexicons.items():
        # res += f"\n\nLEXICON {lexicon}\n\n"
        cur_lexicon = dict(name=lexicon)
        lexicons_res.append(cur_lexicon)
        for morph_dict in morph_list:
            continuation_classes = morph_dict['continuation_class'].split(',')
            affix = morph_dict['affix'] or morph_dict['affix_source']

            gloss = morph_dict['gloss']
            boundary_after = 0 if (morph_dict['no_boundary_after']
                                   or not affix or affix == '0') else 1
            if not boundary_after:
                print(morph_dict)
            if gloss:
                multichars.extend([f"+{gloss_}" for gloss_ in gloss.lstrip('+').split('+')])

            out_mark = morph_dict['out_mark']
            if out_mark:
                multichars.append(out_mark)

            # if bool(gloss) ^ bool(affix):
            #     raise ValueError(f"either none or both of `gloss`, `affix` needed, "
            #                      f"current: {morph_dict}")

            for class_ in continuation_classes:
                # res += (f"{gloss+':' if gloss else ''}{affix if affix else ''}"
                #         f"""{'^ ' if boundary_after else ('' if not affix else ' ')}"""
                #         f"{class_.strip()}{';' if affix else ' ;'}\n")
                cur_lexicon.setdefault('entries', []).append(
                    (f"{gloss + ':' if gloss else ''}"
                     f"{out_mark}{affix if affix else ''}"
                     f"""{'^ ' if boundary_after else ('' if not affix else ' ')}"""
                     f"{class_.strip()}{';' if affix else ' ;'}\n")
                )

        if lexicon == "Root":
            # res += "\n{}\n"
            cur_lexicon.setdefault('entries', []).append("\n{}\n")

    # return res, multichars
    return lexicons_res, multichars


def convert_rules():
    rules = {}
    multichars = []

    rows = read_csv_dict(RULES_FILE)

    categories = {}
    for row in rows:
        categories.setdefault(row['category'], []).append(row)

    for category, rule_list in categories.items():
        if not category:
            continue

        entries = []
        rules[category] = entries
        for rule_dict in rule_list:
            cur_rule = {}
            entries.append(cur_rule)
            name = rule_dict['name']
            if not name:
                continue
            cur_rule['name'] = name
            res = ''

            cur_rule['description'] = rule_dict.get('description')
            # if rule_dict['description']:
                # res += f"# {rule_dict['description']}\n"
            # res += f"define {name} "

            replacement_op = rule_dict['replacement_op'] or '->'
            if rule_dict['is_optional']:
                replacement_op = f"({replacement_op})"

            left_hand_side = rule_dict['structural_desc']
            right_hand_side = rule_dict['structural_change']

            if ',' in left_hand_side:
                left_hand_side_parts = [part.strip() for part in left_hand_side.split(',')]
                right_hand_side_parts = [part.strip() for part in right_hand_side.split(',')]
                if len(left_hand_side_parts) != len(right_hand_side_parts):
                    raise ValueError(f"uneven parts: {left_hand_side_parts}, {right_hand_side_parts}")

                replacement_parts = []
                for left, right in zip(left_hand_side_parts, right_hand_side_parts):
                    replacement_parts.append(f"{add_space(left)} {replacement_op} {add_space(right)}")
                # res = res[:-3]
                # res += " , ".join(replacement_parts)
                cur_rule['replacement'] = " , ".join(replacement_parts)
            else:
                left = left_hand_side
                right = right_hand_side
                # res += f"{add_space(left)} {replacement_op} {add_space(right)}"
                cur_rule['replacement'] = f"{add_space(left)} {replacement_op} {add_space(right)}"

            if '+' in right_hand_side:
                multichars.extend(parse_pluses(right_hand_side))

            left_context = rule_dict['left_context']
            right_context = rule_dict['right_context']
            if not (left_context or right_context):
                # res += ' ;\n\n'
                # cur_rule['entry'] = res
                cur_rule['context'] = None
                continue

            # res += " || "

            if ',' in left_context:
                left_context_parts = [part.strip() for part in left_context.split(',')]
                right_context_parts = [part.strip() for part in right_context.split(',')]
                if len(left_context_parts) != len(right_context_parts):
                    raise ValueError(f"uneven parts: {left_context_parts}, {right_context_parts}")

                context_parts = []
                # res += " , ".join(replacement_parts)
                for left, right in zip(left_context_parts, right_context_parts):
                    # res += f"{add_space(left)} _ {add_space(right)}"
                    context_parts.append(f"{add_space(left)} _ {add_space(right)}")
                # res = res[:-3]
                # res += " , ".join(context_parts)
                cur_rule['context'] = " , ".join(context_parts)
            else:
                # res += f"{left_context} _ {right_context}"
                cur_rule['context'] = f"{left_context} _ {right_context}"

            # res += ' ;\n\n'
            # cur_rule['entry'] = res

    # return res, rule_names, multichars
    harmony = None
    # for category, rules
    NL = '\n'
    processed_rules = {}
    for cat, base_entries in rules.items():
        processed_entries = []
        for rule in base_entries:
            entry = (f"{('# ' + rule['description'] + NL) if rule['description'] else ''}"
                     f"define {rule['name']} {rule['replacement']}"
                     f"{(' || ' + rule['context']) if rule['context'] else ''}"
                     f" ;\n\n"
                     )
            rule['entry'] = entry
            processed_entries.append(rule)
        processed_rules[cat] = processed_entries
    processed_rules.setdefault('technical', []).append(
        dict(name='DeleteLastMorphBoundary',
             entry='define DeleteLastMorphBoundary "^" -> 0 || _ .#. ;')
    )
    return processed_rules, multichars


def make_lexc(multichars_str, morphs_str, lexicon_str, filename="sakha"):
    with open(f'{filename}.lexc', 'w', encoding='utf-8') as f:
        f.write("\n\n".join([
            "!!!sakha.lexc!!!",
            multichars_str,
            morphs_str.format(lexicon_str)]))


def make_foma(definitions_str, rules_str, rule_names, harmony_rule_names,
              affix_specific_rule_names,
              mode="no-guess", guess_substitutes=None, filename="sakha"):
    comment_str = "### sakha.foma ###"

    with open(f'{filename}_{mode}.foma', 'w', encoding='utf-8') as f:
        f.write(f"{comment_str}\n\n")
        f.write(f"{definitions_str}\n\n")
        f.write(f"{rules_str}\n")

        if mode == "no-guess":
            lex_def = "read lexc sakha.lexc\ndefine Lexicon;\n"
            regex_str = "regex Grammar;"
        elif mode == "guess":
            print(guess_substitutes)
            orth_word_rule = "define OrthWord [Consonant* Vowel Consonant*]+;"
            lex_def = (
                "read lexc sakha.lexc\n" +
                '\n'.join(f'substitute defined OrthWord for "{guess_lexicon}"'
                          for guess_lexicon in guess_substitutes) +
                "\ndefine Lexicon;\n"
            )
            regex_str = 'regex [~$["GUESS+"] .o. Grammar] .p. [$["GUESS+"] .o. Grammar];'

            f.write(f"\n\n{orth_word_rule}\n")

        f.write(f"{lex_def}\n")
        f.write(f"define Harmony {harmony_rule_names[0]} .o.\n")
        f.write("\n".join(f"{' ' * 4}{rule} .o." for rule in harmony_rule_names[1:-1]))
        f.write(f"\n{harmony_rule_names[-1]};\n\n\n")

        f.write(f"define Grammar Lexicon .o.\n")
        f.write("\n".join(f"{rule} .o."
            for rule in affix_specific_rule_names + ["Harmony"] * 10 + rule_names[:-1]))
        # f.write(f"Harmony .o.\n")
        # f.write("\n".join(f"{rule} .o." for rule in rule_names[:-1]))
        f.write(f"\n{rule_names[-1]};\n")

        f.write(f"\n{regex_str}\n")


def main(update_lexc=True):
    filename = "sakha"

    morphs, multichars = convert_morph_get_multichars()
    definitions_str = convert_definitions()
    rule_categories, multichars_rules = convert_rules()

    with open(LEXICON_FILE, 'r', encoding='utf-8') as f:
        lexicon_str = f.read()

    guesses_lexicon = re.findall(pos_guess_symb_pat, lexicon_str)
    multichars_lexicon = [GUESS_SYMB] + guesses_lexicon

    print(rule_categories, multichars)

    multichars = {multichar: None
                  for multichar in multichars_lexicon + multichars + multichars_rules}.keys()
    multichars_str = make_multichars_string(multichars)

    print(morphs)

    morphs_str = ''
    for morph in morphs:
        morphs_str += f"\n\nLEXICON {morph['name']}\n\n"
        morphs_str += ''.join(entry for entry in morph['entries'])

    print(morphs_str)

    rules_str = ''.join(f"## {cat}\n\n" + ''.join([rule['entry'] for rule in entries])
                        for cat, entries in rule_categories.items()
                        )
    harmony_rules = rule_categories.pop('harmony')
    affix_specific_rules = rule_categories.pop('affix_specific')
    rule_names = [rule['name']
                  for entries in rule_categories.values()
                  for rule in entries]
    harmony_rule_names = [rule['name']
                          for rule in harmony_rules]
    affix_specific_rule_names = [rule['name']
                                 for rule in affix_specific_rules]

    print(rules_str, harmony_rule_names, rule_names)

    if update_lexc:
        make_lexc(multichars_str, morphs_str, lexicon_str, filename=filename)

    make_foma(definitions_str, rules_str, rule_names, harmony_rule_names,
              affix_specific_rule_names,
              mode="no-guess", filename=filename)

    unique_lexicon_guesses = dict.fromkeys(guesses_lexicon).keys()

    print(multichars_lexicon, unique_lexicon_guesses)
    make_foma(definitions_str, rules_str, rule_names, harmony_rule_names,
              affix_specific_rule_names,
              mode="guess", guess_substitutes=unique_lexicon_guesses,
              filename=filename)


if __name__ == '__main__':
    main(update_lexc=True)


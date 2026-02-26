import csv
import ast
import json
import re
import pandas as pd

CSV_PATH = "/home/nour.rabih/arwi/readability_controlled_generation/generation/prompts_with_6cefr_levels.csv"
# VOCAB_CSV = "/home/nour.rabih/arwi/readability_controlled_generation/generation/vocabs_prompt/5levels/prompts_with_5cefr_levels_and_vocab.csv"
BATCH_INPUT_PATH = "/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_prompt/6levels/batch_input.jsonl"

#open CSV_PATH and VOCAB_CSV and merge them on prompt_id and CEFR_level
prompts = pd.read_csv(CSV_PATH)
# prompts = prompts[prompts['CEFR_level'].isin([ 'C1', 'C2'])]

# prompts.to_csv("/home/nour.rabih/arwi/readability_controlled_generation/generation/C_prompts.csv", index=False)
# CSV_PATH = "/home/nour.rabih/arwi/readability_controlled_generation/generation/C_prompts.csv"
print(f"Loaded {len(prompts)} prompts from {CSV_PATH}")
# vocab = pd.read_csv(VOCAB_CSV)
# merged = pd.merge(prompts, vocab, on=['prompt_id', 'CEFR_level'], how='left')
# save merged to a new CSV
# merged.to_csv('/home/nour.rabih/arwi/generation/vocabs_prompt/3levels/prompts_with_3cefr_levels_and_vocab.csv', index=False)
# CSV_PATH = '/home/nour.rabih/arwi/generation/vocabs_prompt/3levels/prompts_with_3cefr_levels_and_vocab.csv'

# level_map = {
#     1: "A1",
#     2: "B1",
#     3: "B2",
#     4: "C1",
#     5: "C2"
# }

tree_depth_map = {
    "A": (4, 8),
    # "B": (5, 8),
    # "C": (11, 40),
    "A1": (4, 6),
    "A2": (6, 8),
    "B1": (7, 10),
    "B2": (8, 11),
    "C1": (9, 12),
    "C2": (9, 13),
}

def build_prompt(max_depth_low: str, max_depth_high: str, level: str, topic_text: str, profile: dict) -> str: # , words: list[str]




    # kw = "، ".join(words)
    # return ( 
    #     f"اكتب مقالًا إنشائيًا بمستوى قابلية قراءة {level}.\n"
    #     f"المقال يجب أن يكون عن الموضوع التالي: {topic_text}.\n"
    #     "يجب أن يكون المقال مترابطًا، منظمًا (مقدمة، محتوى، وخاتمة)، "
    #     "ومناسبًا لمستوى القراءة المطلوب."
    # )
    # return ( #p2
    #     f"اكتب مقالًا إنشائيًا عربيًا متكاملًا بمستوى قابلية قراءة {level}.\n"
    #     f"يجب أن يتناول المقال الموضوع التالي: {topic_text}.\n"
    #     "يجب أن يكون النص مترابطًا وسلسًا في فقرة أو عدة فقرات متتابعة، "
    #     "ويعكس بنية منطقية تبدأ بتمهيد للفكرة ثم تطويرها ثم إنهائها بخلاصة طبيعية، "
    #     "من دون استخدام عناوين أو كلمات دالة مثل (مقدمة، محتوى، خاتمة) أو أي تقسيم صريح."
    # )
    # return ( # p3
    #     f""" 
    #         اكتب مقالًا إنشائيًا متكاملًا بمستوى قابلية قراءة {level}.
    #         يجب أن يكون المقال عن الموضوع التالي: {topic_text}.
    #         تأكد من إدخال بعض الكلمات المفتاحية التالية بشكل طبيعي داخل النص لإنشاء قصة متماسكة جميلة: {words}.

    #         يجب أن يتكوّن المقال من مقدمة، محتوى، وخاتمة، دون أي عناوين أو تقسيمات ظاهرة.
    #         اكتب النص كسرد متصل في فقرات متتابعة فقط، دون أي تنسيق أو عناوين فرعية.

    #         احرص على أن يكون المقال مترابطًا، منظمًا، ومناسبًا لمستوى القراءة المطلوب.
    #         تأكّد من أن جميع الجمل منطقية ومترابطة من حيث المعنى.
    #         تجنّب أي عبارات أو أفكار غير واقعية أو غير منطقية (مثل مواقف لا تحدث في الواقع).
    #         تأكّد من أن الأمثلة والحقائق تتوافق مع العادات، والثقافة، والمنطق العام.
    #         احرص على أن يكون النص واضحًا وسهل الفهم دون أخطاء لغوية أو تركيبية.
    #     """
    # )

    # return ( #p1 no level
    #     f"اكتب مقالًا إنشائيًا عربيًا متكاملً"
    #     f"يجب أن يتناول المقال الموضوع التالي: {topic_text}.\n"
    #     "يجب أن يكون النص مترابطًا وسلسًا في فقرة أو عدة فقرات متتابعة، "
    #     "ويعكس بنية منطقية تبدأ بتمهيد للفكرة ثم تطويرها ثم إنهائها بخلاصة طبيعية، "
    #     "من دون استخدام عناوين أو كلمات دالة مثل (مقدمة، محتوى، خاتمة) أو أي تقسيم صريح."
    # )

    # return (  # p5 – C-level
    #     f"اكتب مقالًا إنشائيًا عربيًا متكاملًا بمستوى قابلية قراءة {level}.\n"
    #     f"يجب أن يتناول المقال الموضوع التالي: {topic_text}.\n"
    #     "ينبغي أن يكون النص مترابطًا وسلسًا في فقرة واحدة أو عدة فقرات متتابعة، "
    #     "ويعكس بنية فكرية ناضجة تبدأ بتمهيد تحليلي للفكرة، ثم تتوسع في مناقشتها بعمق، "
    #     "مع عرض زوايا متعددة أو وجهات نظر مختلفة عند الاقتضاء، "
    #     "وتنتهي بخلاصة تأملية متوازنة تربط بين الأفكار المطروحة.\n"
    #     "احرص على استخدام تراكيب لغوية متنوعة، وجمل مركبة وطويلة نسبيًا، "
    #     "وروابط منطقية دقيقة تعكس العلاقات السببية والاستنتاجية بين الأفكار، "
    #     "إلى جانب مفردات غنية ومتنوعة تتناسب مع موضوع المقال.\n"
    #     "تجنب التكرار المباشر للأفكار أو الألفاظ، "
    #     "ولا تستخدم عناوين أو كلمات دالة صريحة مثل (مقدمة، محتوى، خاتمة) أو أي تقسيم شكلي."
    # )

    # return (  # p6 – detailed profile syntax + vocab
    #     f"""
    #     اكتب مقالًا إنشائيًا متكاملًا بمستوى قابلية قراءة {level}.
    #     يجب أن يكون المقال عن الموضوع التالي: {topic_text}.
    #     أدرج الكلمات المفتاحية التالية بشكل طبيعي داخل النص: {words}.

    #     حاول أن يكون النص قريبًا من الخصائص العامة التالية:

    #     1) الطول والبنية:
    #     - طول النص قريب من {round(profile["total_words"])}.
    #     - طول الجمل قريب من {round(profile["avg_words_per_sentence"])}.
    #     - عدد الكلمات الفريدة قريب من {round(profile["total_unique_words"])}.
    #     - متوسط طول الكلمة (عدد الحروف): {round(profile["overall_avg_word_len"])}

    #     3) الوصف والتفصيل:
    #     - عدد الصفات : {round(profile["pos_ADJ_mean"])}. 
    #     - استخدم صفات لوصف الأشخاص والأشياء والأفكار.
    #     - كلما زاد المستوى، زادت التفاصيل.

    #     4) بناء الجملة والمعنى:
    #     - عدد الفاعل (قريب من {round(profile["dep_SBJ_mean"])}).
    #     - عدد مفاعيل وأمثلة (قريب من {round(profile["dep_OBJ_mean"])}).
    #     - في المستويات المتقدمة(B1-C2)، اربط السبب بالنتيجة.
    #      5) البنية النحوية (شجرة التحليل):
    #     - العمق الأدنى لشجرة التحليل قريب من {max_depth_low}.
    #     - العمق الأقصى لشجرة التحليل قريب من {max_depth_high}.
    #     - كلما ارتفع المستوى، استخدم جملًا ذات بنية أعمق وأكثر تفرعًا.
        
    #     6) الترابط والمنطق:
    #     - استخدم روابط منطقية تناسب المستوى.
    #     - في المستويات المتقدمة، استخدم التعليل والمقارنة.

    #     7) الأسلوب العام:
    #     - لا أسلوب حواري إلا في B2 أو أعلى.
    #     - تجنّب التعجب والمبالغة في المستويات الدنيا.
    #     - الأسلوب يجب أن يعكس مستوى متعلم حقيقي.

    #     قيود مهمة:
    #     - هذه القيم إرشادية فقط.
    #     - لا تذكر أي أرقام في النص.
    #     - لا تشرح هذه القيود في الناتج.

    #     هيكل النص:
    #     - مقدمة، محتوى، خاتمة بدون عناوين.
    #     - فقرات متصلة فقط.

    #     """
    # )

    return (  # p7 – detailed syntax profile
        f"""
        اكتب مقالًا إنشائيًا متكاملًا بمستوى قابلية قراءة {level}.
        يجب أن يكون المقال عن الموضوع التالي: {topic_text}.

        حاول أن يكون النص قريبًا من الخصائص العامة التالية:

        1) الطول والبنية:
        - طول النص قريب من {round(profile["total_words"])}.
        - طول الجمل قريب من {round(profile["avg_words_per_sentence"])}.
        - عدد الكلمات الفريدة قريب من {round(profile["total_unique_words"])}.
        - متوسط طول الكلمة (عدد الحروف): {round(profile["overall_avg_word_len"])}

        3) الوصف والتفصيل:
        - عدد الصفات : {round(profile["pos_ADJ_mean"])}. 
        - استخدم صفات لوصف الأشخاص والأشياء والأفكار.
        - كلما زاد المستوى، زادت التفاصيل.

        4) بناء الجملة والمعنى:
        - عدد الفاعل (قريب من {round(profile["dep_SBJ_mean"])}).
        - عدد مفاعيل وأمثلة (قريب من {round(profile["dep_OBJ_mean"])}).
        - في المستويات المتقدمة(B1-C2)، اربط السبب بالنتيجة.
         5) البنية النحوية (شجرة التحليل):
        - العمق الأدنى لشجرة التحليل قريب من {max_depth_low}.
        - العمق الأقصى لشجرة التحليل قريب من {max_depth_high}.
        - كلما ارتفع المستوى، استخدم جملًا ذات بنية أعمق وأكثر تفرعًا.
        
        6) الترابط والمنطق:
        - استخدم روابط منطقية تناسب المستوى.
        - في المستويات المتقدمة، استخدم التعليل والمقارنة.

        7) الأسلوب العام:
        - لا أسلوب حواري إلا في B2 أو أعلى.
        - تجنّب التعجب والمبالغة في المستويات الدنيا.
        - الأسلوب يجب أن يعكس مستوى متعلم حقيقي.

        قيود مهمة:
        - هذه القيم إرشادية فقط.
        - لا تذكر أي أرقام في النص.
        - لا تشرح هذه القيود في الناتج.

        هيكل النص:
        - مقدمة، محتوى، خاتمة بدون عناوين.
        - فقرات متصلة فقط.

        """
    )

_nan_token = re.compile(r'(?<![\w])nan(?![\w])', flags=re.IGNORECASE)

def parse_words_cell(s):
    s = str(s).strip()
    # Replace bare nan with None so it's a valid Python literal
    s = _nan_token.sub("None", s)
    return ast.literal_eval(s)

with open(CSV_PATH, "r", encoding="utf-8") as fin, \
     open(BATCH_INPUT_PATH, "w", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    # vocab_reader = csv.DictReader(open(VOCAB_CSV, "r", encoding="utf-8"))
    profiles = pd.read_csv(
        "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/selected_readability_monotonic_features.csv"
    ).set_index("level").to_dict(orient="index")
    # merge vocab by prompt_id and CEFR_level into reader rows
    # vocab_map = {}
    # for row in vocab_reader:
    #     key = (row["prompt_id"], row["CEFR_level"])
        
    #     try:
    #         row["words"] =  parse_words_cell(row["words"])
    #         words = row.get("words", "")
    #         # words = ast.literal_eval(words_raw) if words_raw.strip() else []
    #     except (SyntaxError, ValueError) as e:
    #         # import pdb; pdb.set_trace()
    #         raise ValueError(f"Couldn't parse words={words!r} for prompt_id={row['prompt_id']} and CEFR_level={row['CEFR_level']}") from e
    #     vocab_map[key] = words
    for line_num, row in enumerate(reader, start=1):
        prompt_id = row.get("prompt_id")
        level     = row.get("CEFR_level")
        topic_text = row.get("Arabic Text")
        words_raw = row.get("words", "")
        profile = profiles.get(level)
        if level == "A":
        # the values are the averages across A1 and A2, apply on all columns
            profile = {
                "total_words": int((profiles["A1"]["total_words"] + profiles["A2"]["total_words"]) / 2),
                "avg_words_per_sentence": int((profiles["A1"]["avg_words_per_sentence"] + profiles["A2"]["avg_words_per_sentence"]) / 2),
                "total_unique_words": int((profiles["A1"]["total_unique_words"] + profiles["A2"]["total_unique_words"]) / 2),
                "overall_avg_word_len": round((profiles["A1"]["overall_avg_word_len"] + profiles["A2"]["overall_avg_word_len"]) / 2, 2),
                "pos_ADJ_mean": round((profiles["A1"]["pos_ADJ_mean"] + profiles["A2"]["pos_ADJ_mean"]) / 2, 2),
                "dep_SBJ_mean": round((profiles["A1"]["dep_SBJ_mean"] + profiles["A2"]["dep_SBJ_mean"]) / 2, 2),
                "dep_OBJ_mean": round((profiles["A1"]["dep_OBJ_mean"] + profiles["A2"]["dep_OBJ_mean"]) / 2, 2),
            }

        # words = vocab_map.get((prompt_id, level), [])

        if not prompt_id or not topic_text or not level:
            print(f"[WARN] Skipping line {line_num}: missing prompt_id, level, or Arabic Text")
            continue

        depth_min, depth_max = tree_depth_map[level]
        user_prompt = build_prompt(depth_min, depth_max, level, topic_text, profile) # , words

        # for i in range(10):  # 10 essays per prompt
        custom_id = f"{prompt_id}_{level}_6levels_p7" # _{i+1}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "أنت مساعد يكتب مقالات إنشائية عربية بمستويات قرائية مختلفة."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
        }
        batch_line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        fout.write(json.dumps(batch_line, ensure_ascii=False) + "\n")

print(f"✅ Built batch input file at {BATCH_INPUT_PATH}")

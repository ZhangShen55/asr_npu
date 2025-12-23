import re


_CN_TO_EN = {
    '，': ',',
    '。': '.',
    '！': '!',
    '？': '?',
    '；': ';',
    '：': ':',
    '、': ',',   # 近似成英文里的逗号
    '…': '...',  # 省略号
}


def normalize_punct_by_context(text: str) -> str:
    """
    规则：
    - 只处理“[A-Za-z0-9] + 中文标点”的场景，把中文标点替换成英文标点；
    - 若替换成的英文标点为 , ; : . ? ! 且其后紧跟英文/数字而没有空格，则补一个空格；
    - 小心小数：3.14 这种不强行加空格；
    - 已有空格则不重复添加。
    """
    if not text:
        return text

    # 先处理单字符中文省略号（或模型打出的单个 '…'）
    # 若模型常输出“……”（两个 '…'），本规则也能逐个替换成 '...'，最终出现 '......'，可根据需要后面再收敛为 '...'
    def _repl(m: re.Match) -> str:
        prev = m.group(1)            # 英文/数字
        cn_punct = m.group(2)        # 中文标点
        en_punct = _CN_TO_EN.get(cn_punct, cn_punct)

        end = m.end()
        nxt = text[end] if end < len(text) else ''

        # 是否需要在英文标点后面补空格
        need_space = False
        if en_punct in {',', ';', ':', '?', '!'}:
            # 若下一个非空白是英文/数字，且不是右侧标点/括号/引号，再补一个空格
            if nxt and (not nxt.isspace()) and nxt not in ',.;:!?)}]>"\'':
                need_space = True
        elif en_punct == '.':
            # 小数保护：x.y 且两边都是数字 -> 不加空格
            if prev.isdigit() and nxt.isdigit():
                need_space = False
            else:
                if nxt and (not nxt.isspace()) and nxt not in ',.;:!?)}]>"\'':
                    need_space = True
        elif en_punct == '...':
            # 省略号后一般需要空格（若后面接英文/数字且无空格）
            if nxt and (not nxt.isspace()) and nxt not in ',.;:!?)}]>"\'':
                need_space = True

        return prev + en_punct + (' ' if need_space else '')

    # 1) 英文/数字 + 中文标点 → 调整为英文标点
    new_text = re.sub(r'([A-Za-z0-9])([，。！？；：、…])', _repl, text)

    # 2) 可选：把“……”“………”都收敛成“...”
    new_text = re.sub(r'(?:\.\.\.){2,}', '...', new_text)  # 若多次替换后出现 '......' 等

    # 3) 可选：英文词与英文词之间多空格压成一个（不影响中英文混排）
    new_text = re.sub(r'([A-Za-z0-9])\s{2,}([A-Za-z0-9])', r'\1 \2', new_text)

    return new_text


def _is_cjk(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'

def _is_latin_letter(ch: str) -> bool:
    # 仅 A-Za-z，按你的需求“英文才加空格”
    return ('A' <= ch <= 'Z') or ('a' <= ch <= 'z')

def _needs_space(prev_char: str, next_char: str) -> bool:
    """是否需要在 prev 和 next 之间补空格"""
    if not prev_char or not next_char:
        return False

    # 任何一侧是中文：按你的要求不加空格
    if _is_cjk(prev_char) or _is_cjk(next_char):
        return False

    # 英文缩写/撇号（I'm, it's）：不要加空格
    if next_char in ("'", "’") or prev_char in ("'", "’"):
        return False

    # 连字符（state-of-the-art）两侧不要空格
    if next_char == '-' or prev_char == '-':
        return False

    # 若两侧都是英文字母，则需要补空格
    if _is_latin_letter(prev_char) and _is_latin_letter(next_char):
        return True

    # 可按需扩展：数字与英文之间是否加空格
    # if prev_char.isdigit() and _is_latin_letter(next_char):
    #     return True
    # if _is_latin_letter(prev_char) and next_char.isdigit():
    #     return True

    return False



def safe_concat(prev_text: str, add_text: str) -> str:
    """在分块拼接处做最小必要的空格/空白处理"""
    if not prev_text:
        return add_text or ""
    if not add_text:
        return prev_text

    # 若前一块已以空白结尾，则去掉下一块的前导空白，避免双空格
    if prev_text[-1].isspace():
        add_text = add_text.lstrip()

    # 在“英→英”边界补一个空格
    if _needs_space(prev_text[-1], add_text[0]):
        return prev_text + ' ' + add_text

    # 默认直接拼
    return prev_text + add_text



def count_chinese_and_words(text):
    '''
    计算字符数
    '''
    # 匹配中文汉字的正则表达式
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    # 匹配英文单词的正则表达式
    english_pattern = re.compile(r'[a-zA-Z]+')
    
    # 计算中文汉字数量
    chinese_count = len(chinese_pattern.findall(text))
    # 计算英文单词数量
    english_words = english_pattern.findall(text)
    english_count = len(english_words)
    
    # 返回总字符数
    return chinese_count + english_count
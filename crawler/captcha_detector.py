def is_captcha_page(html: str) -> bool:
    if not html:
        return False

    keywords = [
        "captcha",
        "verify you are human",
        "robot check",
        "unusual traffic",
        "please confirm"
    ]

    html = html.lower()
    return any(k in html for k in keywords)

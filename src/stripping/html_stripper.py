import argparse

from bs4 import BeautifulSoup
from chardet import detect
from lxml import html
from lxml.html.clean import Cleaner


def read_from_disk(path: str):
    with open(path, 'rb') as f:
        encoding = detect(f.read())
    with open(path, 'r', encoding=encoding['encoding']) as f:
        return f.read()


class HtmlStripper:
    def get_text(self, html_content: str):
        cleaner = Cleaner()
        cleaner.style = True
        cleaner.inline_style = True

        cleaned = cleaner.clean_html(html_content)

        soup = BeautifulSoup(cleaned, 'lxml')
        text_lines = soup.findAll(text=True)
        text_lines = [self.strip(line) for line in text_lines if len(self.strip(line)) > 128]
        print(' '.join(text_lines))

    def strip(self, text: str):
        return text.replace('\t', '').replace('\n', '').strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, dest='path', required=True)
    args = parser.parse_args()

    html_content = read_from_disk(args.path)

    tree = html.fromstring(html_content)

    html_stripper = HtmlStripper()
    html_stripper.get_text(html_content)

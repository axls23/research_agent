import zipfile
import sys
import re

def extract_text_from_docx(docx_path, out_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml').decode('utf-8')
            
        texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', xml_content)
        with open(out_path, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(t + '\n')
    except Exception as e:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 2:
        extract_text_from_docx(sys.argv[1], sys.argv[2])

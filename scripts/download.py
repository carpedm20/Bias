"""
Downloads the following:
- Korean Wikipedia texts
- Korean 

"""

from sqlparse import parsestream
from sqlparse.sql import Parenthesis

for statement in parsestream(open('data/test.sql')):
    texts = [str(token.tokens[1].tokens[-1]).decode('string_escape') for token in statement.tokens if isinstance(token, Parenthesis)]
    print texts
    texts = [text for text in texts if text[0] != '#']
    if texts:
        print "\n===\n".join(texts)

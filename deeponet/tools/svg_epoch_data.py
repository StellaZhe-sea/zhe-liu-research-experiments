"""Embed exact numeric epoch records inside SVG metadata without changing curves."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


def embed_epoch_data(path, histories, plotted_metric=None):
    path=Path(path)
    records={}
    for method,rows in histories.items():
        fields=list(rows[0])
        records[method]={'columns':fields,
                         'values':[[float(row[k]) for k in fields] for row in rows]}
    payload={'schema':'deeponet-every-epoch-v1','smoothing':False,'subsampling':False,
             'plotted_metric':plotted_metric,'records':records}
    raw=json.dumps(payload,separators=(',',':'),allow_nan=False)
    content=path.read_text(encoding='utf-8')
    # This helper runs immediately after a fresh Matplotlib SVG save.
    assert 'id="epoch-data"' not in content
    content=content.replace('</svg>','<metadata id="epoch-data">'+escape(raw)+'</metadata>\n</svg>')
    path.write_text(content,encoding='utf-8')
    root=ET.parse(path).getroot()
    node=root.find("{http://www.w3.org/2000/svg}metadata[@id='epoch-data']")
    assert node is not None and json.loads(node.text)==payload
    return payload

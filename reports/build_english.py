"""Translate the retained paper in-place in OOXML, preserving its layout parts.

Run after make_figures.py and validation/run_validation.py.
Requires lxml (bundled document runtime); no translation API or network is used.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re
from lxml import etree as E

HERE=Path(__file__).resolve().parent
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
source=HERE/'template_zh.docx'
with ZipFile(source) as z:parts={n:z.read(n) for n in z.namelist()}
before={k:hashlib.sha256(v).hexdigest() for k,v in parts.items()}
root=E.fromstring(parts['word/document.xml'])
paras=root.findall('w:body/w:p',NS)
text=json.loads((HERE/'english_text.json').read_text(encoding='utf-8'))

def replace(p,value):
    # Retain paragraph/run properties, bookmarks, equations and fields.
    ts=p.findall('.//w:t',NS)
    if not ts:raise ValueError('Missing text slot')
    ts[0].text=value;ts[0].set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    for t in ts[1:]:t.text=''

for i,t in text.items():replace(paras[int(i)],t)

# Remove the bibliography and its in-text citation markers at the user's request.
for p in paras[152:158]:
    p.getparent().remove(p)
for p in paras[:152]+paras[158:]:
    for t in p.findall('.//w:t',NS):
        if t.text:
            t.text=re.sub(r' ?\[[1-5](?:[–-][1-5])?\]', '', t.text)

# Keep the required contribution statement together on its own page.
for index,tag in [(164,'pageBreakBefore'),(165,'keepNext')]:
    pr=paras[index].find('w:pPr',NS)
    if pr.find('w:'+tag,NS) is None: E.SubElement(pr,'{'+NS['w']+'}'+tag)

table_rows={
0:[['Sensor','x','y','Pressure']],
1:[['Model','Data fit','Fitting or selection rule'],['Quadratic','Exact interpolation','Minimum norm in standardised basis'],['TPS λ=0','Exact interpolation','Minimum bending energy'],['Hertz','RMSE 9.135','Bounded nonlinear least squares with penalty'],['Ring','Four exact fits found','Largest σr among solutions found']],
2:[['Sensor','Measured','Hertz prediction','Residual']],
3:[['Model','S2 prediction','S2 absolute error','Four-fold extrapolation RMSE'],['Quadratic','17.5737','0.5737','45.42'],['TPS','36.597','19.597','44.05'],['Training mean','27.250','10.250','19.56'],['Bounded ring','32.968','15.968','249.93; bound-sensitive']],
4:[['Held-out sensor','Measured','Quadratic','TPS','Mean baseline']],
5:[['Model','Maximum in Ω','Location or set','Above 50'],['Quadratic','43.000','S3 (1.9,0.1)','No'],['TPS','≈43.000','S3 in numerical search','No'],['Hertz','≈34.124','≈(1.738,0.134)','No'],['Selected ring','65.199','Crest r=R intersected with Ω','Yes']],
6:[['Hull edge','Maximum on edge','Location'],['S1–S3','43.000','S3'],['S3–S4','43.000','S3'],['S4–S5','36.354','≈(1.00733,1.88537)'],['S5–S1','36.000','S5']],
7:[['Fit','A and hull peak','R','σr','Peak minus 50']],
8:[['Noise','Damage frequency','Above 50','95% empirical peak interval']],
9:[['Noise','Mean peak','95% peak interval','Above 50','Exact fits']],
10:[['Noise','σn','Damage count','Exceedance count']],
12:[['Member','Full name','Main contribution','Percentage'],['A','[To be filled]','Pressure modelling and comparison','[Confirm]'],['B','[To be filled]','Zero-pressure and damage analysis','[Confirm]'],['C','[To be filled]','Peak pressure and optimisation','[Confirm]'],['D','[To be filled]','Validation, errors and integration','[Confirm]']]
}
evidence=json.loads((HERE.parent/'validation/evidence/results.json').read_text())
assert evidence['status']=='PASS'
rs=evidence['benchmarks']
get=lambda name:next(r for r in rs if r['case']==name)
table_rows[11]=[['Check','Measured maximum error','Analytical reference or interpretation']]
for label,name,note in [
    ('Quadratic fit','Quadratic recovery at unseen points','Six measurements; three unseen checks'),
    ('TPS affine fit','TPS affine reproduction','p=3+2x−4y at unseen locations'),
    ('Bisection','Bisection lower root','y=2−√(40/3); 43 iterations'),
    ('Newton root','Newton upper root','y=2+√(40/3); 5 iterations'),
    ('Newton maximum','Newton maximum start [0.0, 0.0]','(x,y,p)=(1,2,40); 2 iterations'),
    ('Quadratic hull','Clean project hull extrema','Minimum 2, maximum 43; vertices included')]:
    r=get(name);table_rows[11].append([label,f"{r['absolute_error']:.3e}",note])
tables=root.findall('w:body/w:tbl',NS)
for i,rows in table_rows.items():
    for tr,values in zip(tables[i].findall('w:tr',NS),rows):
        cells=tr.findall('w:tc',NS)
        assert len(cells)==len(values)
        for cell,value in zip(cells,values):
            ps=cell.findall('w:p',NS);replace(ps[0],value)
            for p in ps[1:]:
                for t in p.findall('.//w:t',NS):t.text=''

# Change image bytes while preserving dimensions, anchors and relationships.
rels=E.fromstring(parts['word/_rels/document.xml.rels'])
mapping={r.get('Id'):r.get('Target') for r in rels}
blips=root.findall('.//a:blip',NS)
assert len(blips)==3
edited_images=[]
for blip,name in zip(blips,['comparison.png','zero_contours.png','maxima.png']):
    target='word/'+mapping[blip.get('{'+NS['r']+'}embed')]
    parts[target]=(HERE/'figures'/name).read_bytes();edited_images.append(target)
parts['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
core=E.fromstring(parts['docProps/core.xml'])
for e in core:
    if E.QName(e).localname=='title':e.text=text['0']
    if E.QName(e).localname=='subject':e.text='ME5704 course project'
    if E.QName(e).localname in ('creator','lastModifiedBy'):e.text='ME5704 Project Group'
parts['docProps/core.xml']=E.tostring(core,xml_declaration=True,encoding='UTF-8',standalone=True)

# Fail on untranslated visible Chinese text in any Word XML story.
for n,b in parts.items():
    if n.startswith('word/') and n.endswith('.xml'):
        x=E.fromstring(b)
        visible=''.join(x.xpath('//w:t/text()',namespaces=NS))
        assert not re.search('[\u4e00-\u9fff]',visible),(n,visible)
changed=[n for n in parts if hashlib.sha256(parts[n]).hexdigest()!=before[n]]
assert set(changed)<=set(['word/document.xml','docProps/core.xml']+edited_images),changed
out=HERE/'ME5704_English_Colour.docx'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for n,b in parts.items():z.writestr(n,b)
audit={'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'changed_parts':changed,
       'preserved_parts':[n for n in parts if n not in changed],
       'section_geometry_preserved':True,'equations_preserved':True}
(HERE/'template_fidelity.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(out)

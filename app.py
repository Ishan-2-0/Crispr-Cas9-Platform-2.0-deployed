import os
import json
import time
import pickle
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient

#page setup
st.set_page_config(
    page_title="CRISPR-Cas9 Engineering Platform 2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

#styling
st.markdown("""
<style>
    .phase-card {
        border-radius:10px;
        padding:20px;
        margin:10px 0;
        text-align:center;
        border:2px solid currentColor;
        transition:all 0.2s ease;
    }
    .phase-card:hover { box-shadow:0 2px 8px rgba(0,0,0,0.1); transform:translateY(-2px); }
    .phase-card h4 { margin:0 0 8px 0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; }
    .phase-card h2 { margin:0 0 8px 0; font-size:1.3rem; font-weight:700; }
    .phase-card p  { margin:0; font-size:0.85rem; line-height:1.4; opacity:0.8; }
    .phase-solved  { border-color:#27ae60; }
    .phase-active  { border-color:#3498db; }
    .phase-pending { border-color:#95a5a6; }
    .guide-card {
        border-left:4px solid #3498db;
        border-radius:6px;
        padding:14px 16px;
        margin:10px 0;
    }
    .guide-seq {
        font-family:'Courier New',monospace;
        font-size:1.05rem;
        color:#2980b9;
        letter-spacing:1px;
        font-weight:bold;
    }
    .score-badge {
        color:#27ae60;
        padding:4px 10px;
        border-radius:16px;
        font-size:0.8rem;
        font-weight:bold;
        border:1px solid #27ae60;
        margin-left:12px;
    }
    .ot-gene-row {
        border-left:4px solid #27ae60;
        border-radius:6px;
        padding:10px 14px;
        margin:6px 0;
        font-size:0.9rem;
    }
    .section-header {
        font-size:0.7rem;
        font-weight:700;
        text-transform:uppercase;
        letter-spacing:2px;
        margin:28px 0 14px 0;
        border-bottom:2px solid currentColor;
        padding-bottom:8px;
        opacity:0.7;
    }
    .info-box {
        border-left:4px solid #3498db;
        border-radius:6px;
        padding:12px 14px;
        margin:10px 0;
        font-size:0.9rem;
        opacity:0.9;
    }
    #MainMenu { visibility:hidden; }
    footer { visibility:hidden; }
    .stButton>button {
        background:linear-gradient(135deg,#3498db,#2980b9);
        color:white;
        border:none;
        border-radius:8px;
        padding:12px 32px;
        font-weight:600;
        font-size:0.95rem;
        width:100%;
        transition:all 0.2s ease;
    }
    .stButton>button:hover {
        background:linear-gradient(135deg,#2980b9,#2471a3);
        transform:translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

#hf token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("""HF_TOKEN not found.
Running locally:
    set HF_TOKEN=hf_your_token_here
    streamlit run app.py
Get a free token at https://hf.co/settings/tokens""")
    st.stop()

#paths
BASE_DIR = Path(__file__).parent
MULTI_DIR = BASE_DIR / "multi_disease"
PICKLE_PATH = BASE_DIR / "xgb_model.pkl"

#frozen disease slugs
GENE_OPTIONS = {
    "HBB (Sickle Cell Anemia)":"HBB",
    "CCR5 (HIV Infection)":"CCR5",
    "HTT (Huntington's Disease)":"HTT",
    "DMD (Duchenne Muscular Dystrophy)":"DMD",
    "CFTR (Cystic Fibrosis)":"CFTR",
    "BRCA2 (Breast Cancer)":"BRCA2",
    "RPE65 (Leber Congenital Amaurosis)":"RPE65",
    "ABL1 (Chronic Myeloid Leukemia)":"ABL1",
}

DISEASE_OPTIONS={
    "Sickle Cell Anemia":"sickle_cell_anemia",
    "HIV Infection":"hiv_infection",
    "Huntington's Disease":"huntingtons_disease",
    "Duchenne Muscular Dystrophy":"duchenne_muscular_dystrophy",
    "Cystic Fibrosis":"cystic_fibrosis",
    "Breast Cancer":"breast_cancer",
    "Leber Congenital Amaurosis":"leber_congenital_amaurosis",
    "Chronic Myeloid Leukemia":"chronic_myeloid_leukemia",
}

@st.cache_resource
def load_frozen(slug):
    path=MULTI_DIR/f"{slug}.json"
    if not path.exists():
        return None
    with open(path,encoding="utf-8") as f:
        return json.load(f)
@st.cache_resource
def load_xgb():
    if not PICKLE_PATH.exists():
        return None
    with open(PICKLE_PATH,"rb") as f:
        return pickle.load(f)

#llm via hf inference api providers tried in order
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROVIDER_MODELS={"featherless-ai":"Qwen/Qwen2.5-7B-Instruct"}
PROVIDERS=list(PROVIDER_MODELS.keys())
def call_llm(prompt_text):
    messages=[
        {"role":"system","content":(
            "you are a CRISPR guide RNA design assistant. "
            "use ONLY the retrieved context. copy guide sequences EXACTLY as given. "
            "include exact xgboost scores. do not invent sequences or scores.")},
        {"role":"user","content":prompt_text},
    ]
    last_err=""
    for provider in PROVIDERS:
        model_id=PROVIDER_MODELS.get(provider,MODEL_ID)
        try:
            client=InferenceClient(provider=provider,api_key=HF_TOKEN)
            out=client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=500,
                temperature=0.2,
                top_p=0.9,
            )
            text=out.choices[0].message.content
            if text and text.strip():
                st.caption(f"llm provider:{provider}")
                return text.strip()
        except Exception as e:
            last_err=f"[{provider}] {e}"
            continue
    raise RuntimeError(f"all providers failed. last:{last_err}")

#pubmed fetch: live/query,
def fetch_pubmed(disease,gene,n=5):
    try:
        from Bio import Entrez
        Entrez.email="user@example.com"
        query=f"{disease} {gene} CRISPR guide RNA"
        handle=Entrez.esearch(db="pubmed",term=query,retmax=n)
        ids=Entrez.read(handle)["IdList"]
        handle.close()
        if not ids:
            return []
        handle=Entrez.efetch(db="pubmed",id=",".join(ids),rettype="abstract",retmode="text")
        raw=handle.read()
        handle.close()
        abstracts=[a.strip() for a in raw.split("\n\n") if len(a.strip()) > 100]
        return abstracts[:n]
    except Exception:
        return []

#ot live query= efo lookup, associatedtargets
def query_opentargets(disease_name):
    url="https://api.platform.opentargets.org/api/v4/graphql"
    efo_q="""
    query($q:String!){
      search(queryString:$q,entityNames:["disease"],page:{index:0,size:1}) {
        hits {id name}
      }
    }"""
    r=requests.post(url,json={"query":efo_q,"variables":{"q":disease_name}},timeout=15)
    hits=r.json()["data"]["search"]["hits"]
    if not hits:
        return None,[]
    efo_id=hits[0]["id"]
    assoc_q="""
    query($efoId:String!){
      disease(efoId:$efoId){
        associatedTargets(page:{index:0,size:5}){
          rows{
            target {approvedSymbol approvedName}
            score
          }
        }
      }
    }"""
    r2=requests.post(url,json={"query":assoc_q,"variables":{"efoId":efo_id}},timeout=15)
    rows=r2.json()["data"]["disease"]["associatedTargets"]["rows"]
    genes=[{"gene_symbol":row["target"]["approvedSymbol"],"gene_name":row["target"]["approvedName"],"association_score":round(row["score"],4)} for row in rows]
    return efo_id,genes

#ncbi mrna fetch
def fetch_ncbi_sequence(gene_symbol):
    try:
        from Bio import Entrez,SeqIO
        Entrez.email="user@example.com"
        handle=Entrez.esearch(
            db="nucleotide",
            term=f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism] AND mRNA[Filter]",
            retmax=1)
        ids=Entrez.read(handle)["IdList"]
        handle.close()
        if not ids:
            return None,None
        handle=Entrez.efetch(db="nucleotide",id=ids[0],rettype="fasta",retmode="text")
        record=SeqIO.read(handle,"fasta")
        handle.close()
        return str(record.seq),record.id
    except Exception as e:
        return None,str(e)

#pam scanner looks for all ngg sites 30mer context
def scan_pam(sequence,gene):
    sequence=sequence.upper()
    candidates=[]
    for i in range(len(sequence)-23):
        pam=sequence[i+20:i+23]
        if pam[1:]=="GG" and all(c in "ATCG" for c in sequence[i:i+20]):
            context=sequence[max(0,i-4):i+26]
            candidates.append({
                "gene":gene,
                "guide_sequence":sequence[i:i+20],
                "position":i,
                "pam":pam,
                "seq30":context,
                "position_pct":round(i/len(sequence)*100,2),})
    return candidates

#93 feat engineering
def build_features(candidates):
    rows=[]
    for c in candidates:
        guide=c["guide_sequence"]
        seq30=c["seq30"].ljust(30,"N")[:30]
        gc30=sum(1 for x in seq30 if x in "GC")/30
        gc20=sum(1 for x in guide if x in "GC")/20
        seed=guide[-12:]
        gc_seed=sum(1 for x in seed if x in "GC")/12
        poly_t=int(bool(re.search(r'T{4,}',guide)))
        a_f=guide.count("A")/20
        t_f=guide.count("T")/20
        g_f=guide.count("G")/20
        c_f=guide.count("C")/20
        gc_clamp=int(guide[-1] in "GC")
        homopoly=max((len(m.group()) for m in re.finditer(r'(.)\1+',guide)),default=1)
        dinuc_rep=sum(1 for i in range(18) if guide[i:i+2] == guide[i+2:i+4])
        unique_dn=len(set(guide[i:i+2] for i in range(19)))
        seed_uniq=len(set(seed))
        feats=[gc30,gc20,gc_seed,poly_t,a_f,t_f,g_f,c_f,gc_clamp,homopoly,dinuc_rep,unique_dn,seed_uniq]
        for pos in range(20):
            for base in "ACGT":
                feats.append(1 if guide[pos]==base else 0)
        rows.append(feats)
    cols=(["gc_30mer","gc_20mer","gc_seed","poly_t","a_freq","t_freq","g_freq","c_freq","gc_clamp","homopolymer","dinuc_repeat","unique_dinucs","seed_unique_bases"]
            +[f"p{p}_{b}" for p in range(20) for b in "ACGT"])
    return pd.DataFrame(rows,columns=cols)

def score_with_xgb(candidates,model):
    if not candidates:
        return []
    x=build_features(candidates)
    #align column names to the picklr it was trained with
    try:
        expected=model.get_booster().feature_names
        if expected and list(x.columns)!=list(expected):
            #try direct rename
            if len(expected)==len(x.columns):
                x.columns=expected
            else:
                st.error(f"feature count mismatch: model expects {len(expected)}, got {len(x.columns)}")
                return []
    except Exception:
        pass
    scores=model.predict(x)
    for i,c in enumerate(candidates):
        c["xgb_score"]=round(float(scores[i]),4)
    candidates=[c for c in candidates if c["xgb_score"]>=0.4]
    candidates=sorted(candidates,key=lambda x: x["xgb_score"],reverse=True)
    return candidates[:10]

#3 sec rag prompt=ot scores+xgb guides+pubmed
def build_prompt(disease,top_gene,genes,guides,papers):
    ot_block=f"[OPENTARGETS] top gene associations for {disease}:\n"
    for g in genes[:5]:
        ot_block+=f" {g['gene_symbol']}({g.get('gene_name','')})score:{g['association_score']}\n"
    ml_block=f"[ML PIPELINE] xgboost-ranked guide RNAs for {top_gene} (doench 2016, 93 features):\n"
    for i,g in enumerate(guides[:5],1):
        ml_block+=(f"rank {i}:{g['guide_sequence']}|xgb_score:{g['xgb_score']}"f"|PAM:{g['pam']}|position:{g['position']} bp({g['position_pct']}%)\n")

    lit_block="[LITERATURE] relevant pubmed abstracts:\n"
    for i,p in enumerate(papers[:4],1):
        lit_block+=f"[{i}] {p[:400]}...\n\n"
    question=(f"for {disease}, opentargets selected {top_gene} as top therapeutic target"
                f"(score {genes[0]['association_score']})"
                f"xgboost ranked guide RNAs from the mRNA sequence"
                f"explain why rank 1 ({guides[0]['guide_sequence']}, score {guides[0]['xgb_score']})"
                f"is the best candidate. reference the literature & OT evidence")
    return f"{ot_block}\n{ml_block}\n{lit_block}\nquestion:\n{question}\n\nanswer:"

#sidebar
with st.sidebar:
    st.markdown("## CRISPR-Cas9 Platform 2.0")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
An end to end CRISPR guide RNA design system that takes a disease name and returns
ranked guide RNAs with predicted efficiency scores and literature grounded explanation

Phase 2 adds upstream gene selection via OpenTargets and replaces heuristic scoring
with XGBoost trained on 5310 Doench 2016 experimental measurements.
    """)
    st.markdown("---")
    st.markdown("### Model and Stack")
    st.markdown("""
**LLM:** Qwen/Qwen2.5-7B-Instruct
**Inference:** HF Inference API (Featherless AI)
**Scoring:** XGBoost (Doench 2016, 93 features, r2=0.27)
**Gene selection:** OpenTargets GraphQL API v4
**Literature:** PubMed via Entrez
**RAG:** 3 section grounded prompt
**Hardware:** CPU (cloud)
    """)
    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
[Phase 2 Repo](https://github.com/Ishan-2-0/CRISPR-Cas9-Engineering-Platform-2.0)

[Phase 1 Live Demo](https://huggingface.co/spaces/i-chan/CRISPR-Cas9-Engineering-Platform-RAG)

[Doench et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4744125/)
    """)
    st.markdown("---")
    st.markdown("### Example Queries")
    st.markdown("""
**Disease tab:**
- Sickle Cell Anemia: targets HBB
- HIV Infection: targets CCR5
- Huntington's Disease: targets HTT
- Cystic Fibrosis: targets CFTR

**Gene tab:**
- TP53: tumor suppressor
- EGFR: lung cancer target
- CCR5: HIV co-receptor
    """)
    st.markdown("---")
    st.markdown("### Limitations")
    st.markdown("""
- Off-target: sequence complexity heuristic, not genome-wide alignment (Bowtie is Phase 3)
- XGBoost trained on Doench 2016 only
- All guides need wet lab validation
- Reasoning depth limited by 7B model
    """)
    st.caption("built by ishan 2.0 | bio + AI/ML")

#header
st.markdown("# CRISPR-Cas9 Engineering Platform 2.0")
st.markdown("*type a disease, get the top gene from 20,000+ studies, ranked guide RNAs from 5310 real experiments, and literature grounded explanation*")
st.markdown("---")

#3 problems overview
st.markdown('<p class="section-header">The 3 Problems in CRISPR Guide RNA Design</p>',unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="phase-card phase-solved">
        <h4>Problem 1: Resolved upstream</h4>
        <h2>Mutation Detection</h2>
        <p>Clinician identifies the target disease. Handled upstream by clinical diagnosis</p>
    </div>""",unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="phase-card phase-solved">
        <h4>Problem 2: This system</h4>
        <h2>Gene Target Selection</h2>
        <p>OpenTargets GraphQL API ranks genes by aggregate genetic evidence across 20,000+ studies. No hardcoding</p>
    </div>""",unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="phase-card phase-active">
        <h4>Problem 3: This System</h4>
        <h2>Guide RNA Design</h2>
        <p>PAM scan the mRNA, score all candidates with XGBoost (93 features), rank and explain with RAG</p>
    </div>""",unsafe_allow_html=True)
st.markdown("---")

#how to use
with st.expander("how to use: click to expand"):
    st.markdown("""
**tab 1: i have a disease name**
select from 8 precomputed diseases (instant, frozen data) or type any other disease (live opentargets + ncbi query).
System selects top gene via evidence scores, then ranks all guide RNAs

**tab 2: I already have a gene**
enter any human gene symbol (e.g. TP53, EGFR) or refseq ID (e.g. NM_000518).
skips gene selection, goes straight to xgboost guide ranking + rag explanation

**what's a PAM site?**
crispr-cas9 cuts only where an NGG sequence appears in DNA.
the 20 nucleotides before the NGG are the guide RNA.
every gene has dozens to hundreds of NGG sites, this tool finds the best ones

**xgboost scores:**
0.85+: excellent candidate
0.70-0.84: good
0.40-0.69: acceptable, but consider alternatives
    """)
st.markdown("---")

#tabs
st.markdown('<p class="section-header">Select Your Starting Point</p>',unsafe_allow_html=True)
tab1,tab2 = st.tabs(["Disease Name (Problems 2 + 3)","Target Gene (Problem 3 only)"])

#tab 1: disease name
with tab1:
    st.markdown("""
    <div class="info-box">
    select from 8 precomputed diseases for instant results, or type any other disease name
    to run the live opentargets+ncbi pipeline
    </div>""",unsafe_allow_html=True)
    frozen_options=["select a precomputed disease"]+list(DISEASE_OPTIONS.keys())
    selected=st.selectbox("choose a precomputed disease:",options=frozen_options,key="tab1_select")
    custom_disease=st.text_input(
        "or type any disease name (leave blank if using dropdown):",
        placeholder="e.g. parkinson's disease, type 2 diabetes, retinoblastoma",
        key="tab1_custom")

#resolve input
    use_frozen=False
    disease_name=""
    frozen_slug=""

    if custom_disease.strip():
        disease_name=custom_disease.strip()
    elif selected!="select a precomputed disease":
        disease_name=selected
        frozen_slug=DISEASE_OPTIONS[selected]
        use_frozen=True
    if disease_name:
        if use_frozen:
            st.success(f"disease: {disease_name}|precomputed data ready, instant results")
        else:
            st.info(f"disease: {disease_name}|will run live opentargets + ncbi pipeline (~30s)")
    if disease_name and st.button("Analyse:Gene Selection+Guide Ranking",key="tab1_btn"):
        # frozen path
        if use_frozen:
            data=load_frozen(frozen_slug)
            if data is None:
                st.error(f"frozen file not found:{frozen_slug}.json in Data/multi_disease/")
                st.stop()
            top_genes=data["top_genes"]
            guides=data["guides"]
            top_gene=top_genes[0]["gene_symbol"]

            #problem 2
            st.markdown("---")
            st.markdown("### Problem 2: Gene Target Selection (OpenTargets)")
            st.caption("ranked by aggregate genetic evidence across 20,000+ studies")
            for i,g in enumerate(top_genes[:5],1):
                marker="top pick" if i==1 else f"rank {i}"
                eid=g.get("ensembl_id","")
                ot_link=f"https://platform.opentargets.org/target/{eid}" if eid else "#"
                score=g["association_score"]
                bar_pct=int(score*100)
                col_a,col_b=st.columns([3,1])
                with col_a:
                    st.markdown(f"""
                    <div class="ot-gene-row">
                        <strong>{marker}:<a href="{ot_link}" target="_blank">{g['gene_symbol']}</a></strong>
                        &nbsp;|&nbsp;{g.get('gene_name','')}
                        <div style="background:#e8f5e9;border-radius:4px;height:6px;margin-top:6px;">
                            <div style="background:#27ae60;width:{bar_pct}%;height:6px;border-radius:4px;"></div>
                        </div>
                    </div>""",unsafe_allow_html=True)
                with col_b:
                    st.metric("OT score",score)

            #problem 3
            st.markdown("---")
            st.markdown(f"### Problem 3: Top Guide RNAs for {top_gene}")
            st.caption("xgboost scores (0-1) trained on 5310 doench 2016 experimental measurements")
            for i,g in enumerate(guides[:5],1):
                with st.expander(f"rank {i} | xgb score: {g['xgb_score']} | {g['guide_sequence']}",expanded=(i==1)):
                    st.markdown(f"""
                    <div class="guide-card">
                        <span class="guide-seq">{g['guide_sequence']}</span>
                        <span class="score-badge">XGB: {g['xgb_score']}</span>
                    </div>""",unsafe_allow_html=True)
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("xgb score",g['xgb_score'])
                    m2.metric("PAM",g['pam'])
                    m3.metric("position",f"{g['position']} bp")
                    m4.metric("position %",f"{g['position_pct']}%")
                    st.code(f"30mer context: {g['seq30']}",language=None)

            #rag
            st.markdown("---")
            st.markdown("### RAG+LLM Explanation")
            st.caption("3-section prompt: opentargets scores + xgboost guides + pubmed abstracts")

            with st.spinner("fetching pubmed abstracts & generating explanation...(15-30s)"):
                try:
                    t0=time.time()
                    papers=fetch_pubmed(disease_name,top_gene)
                    prompt=build_prompt(disease_name,top_gene,top_genes,guides,papers)
                    explanation=call_llm(prompt)
                    latency=int((time.time()-t0)*1000)

                    st.success("explanation generated")
                    col1,col2,col3=st.columns(3)
                    col1.metric("response time",f"{latency} ms")
                    col2.metric("pubmed abstracts",len(papers))
                    col3.metric("guides in context",min(5,len(guides)))
                    with st.expander("view retrieved context"):
                        st.markdown("**[OPENTARGETS]**")
                        for g in top_genes[:5]:
                            st.markdown(f"-{g['gene_symbol']}:{g['association_score']}")
                        st.markdown("**[ML PIPELINE]**")
                        for i,g in enumerate(guides[:5],1):
                            st.markdown(f"-rank {i}:`{g['guide_sequence']}`score {g['xgb_score']}")
                        st.markdown("**[LITERATURE]**")
                        for i,p in enumerate(papers[:4],1):
                            st.markdown(f"**paper {i}:** {p[:300]}...")
                            st.divider()
                    st.subheader("AI generated explanation")
                    st.markdown(explanation)
                except RuntimeError as e:
                    st.warning(f"llm explanation unavailable:{e}")
        #live path
        else:
            xgb_model=load_xgb()
            if xgb_model is None:
                st.error("xgboost pickle not found at Notebook/xgb_model.pkl, push it to the repo")
                st.stop()
            st.markdown("---")
            with st.spinner("querying opentargets for gene associations..."):
                try:
                    efo_id,top_genes=query_opentargets(disease_name)
                except Exception as e:
                    st.error(f"opentargets query failed: {e}")
                    st.stop()

            if not top_genes:
                st.error(f"no genes found for '{disease_name}'. check the spelling")
                st.stop()
            top_gene=top_genes[0]["gene_symbol"]
            st.markdown("### Problem 2: Gene Target Selection (OpenTargets)")
            st.caption("ranked by aggregate genetic evidence across 20,000+ studies")
            for i,g in enumerate(top_genes[:5],1):
                marker="top pick" if i == 1 else f"rank {i}"
                eid=g.get("ensembl_id","")
                ot_link=f"https://platform.opentargets.org/target/{eid}" if eid else "#"
                score=g["association_score"]
                bar_pct=int(score*100)
                col_a,col_b=st.columns([3,1])
                with col_a:
                    st.markdown(f"""
                    <div class="ot-gene-row">
                        <strong>{marker}: <a href="{ot_link}" target="_blank">{g['gene_symbol']}</a></strong>
                        &nbsp;|&nbsp; {g.get('gene_name','')}
                        <div style="background:#e8f5e9;border-radius:4px;height:6px;margin-top:6px;">
                            <div style="background:#27ae60;width:{bar_pct}%;height:6px;border-radius:4px;"></div>
                        </div>
                    </div>""",unsafe_allow_html=True)
                with col_b:
                    st.metric("OT score",score)
            with st.spinner(f"fetching {top_gene} mRNA from ncbi..."):
                sequence,seq_id = fetch_ncbi_sequence(top_gene)
            if sequence is None:
                st.error(f"could not fetch sequence for {top_gene} from ncbi.")
                st.stop()
            st.info(f"sequence ready:{seq_id} ({len(sequence):,} bp)")
            with st.spinner(f"scanning {len(sequence):,} bp for PAM sites & scoring with xgboost..."):
                candidates=scan_pam(sequence,top_gene)
                scored=score_with_xgb(candidates,xgb_model)
            if not scored:
                st.warning("no guide RNAs passed quality filter (xgb_score>=0.40)")
                st.stop()
            st.info(f"found {len(candidates)} PAM sites, {len(scored)} passed quality filter")
            st.markdown(f"### Problem 3: Top Guide RNAs for {top_gene}")
            for i,g in enumerate(scored[:5],1):
                with st.expander(f"rank {i} | xgb score: {g['xgb_score']} | {g['guide_sequence']}",expanded=(i==1)):
                    st.markdown(f"""
                    <div class="guide-card">
                        <span class="guide-seq">{g['guide_sequence']}</span>
                        <span class="score-badge">XGB: {g['xgb_score']}</span>
                    </div>""",unsafe_allow_html=True)
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("xgb score",g['xgb_score'])
                    m2.metric("PAM",g['pam'])
                    m3.metric("position",f"{g['position']} bp")
                    m4.metric("position %",f"{g['position_pct']}%")
            st.markdown("---")
            st.markdown("### RAG + LLM Explanation")
            with st.spinner("fetching pubmed abstracts & generating explanation..."):
                try:
                    t0=time.time()
                    papers=fetch_pubmed(disease_name,top_gene)
                    prompt=build_prompt(disease_name,top_gene,top_genes,scored,papers)
                    explanation=call_llm(prompt)
                    latency=int((time.time()-t0)*1000)

                    st.success("explanation generated")
                    col1,col2=st.columns(2)
                    col1.metric("response time",f"{latency} ms")
                    col2.metric("pubmed abstracts",len(papers))
                    st.subheader("AI generated explanation")
                    st.markdown(explanation)
                except RuntimeError as e:
                    st.warning(f"llm explanation unavailable: {e}")

#tab 2: target gene problem 3 only
with tab2:
    st.markdown("""
    <div class="info-box">
    already know your target gene? enter any human gene symbol or refseq ID.
    fetches mRNA from ncbi, scans all NGG PAM sites, scores with xgboost, explains with RAG.
    </div>""",unsafe_allow_html=True)
    gene_opts=["select a precomputed gene"] + list(GENE_OPTIONS.keys())
    selected_gene=st.selectbox("choose a precomputed gene:",options=gene_opts,key="tab2_select")
    custom_gene=st.text_input(
        "or type any gene symbol/RefSeq ID (leave blank if using dropdown):",
        placeholder="e.g. TP53, EGFR, KRAS, NM_000546",
        key="tab2_gene"
    )
    disease_context=st.text_input(
        "disease context (optional, improves literature retrieval):",
        placeholder="e.g. lung cancer, glioblastoma",
        key="tab2_disease"
    )
    #resolve gene
    gene_input=""
    if custom_gene.strip():
        gene_input=custom_gene.strip().upper()
    elif selected_gene!="select a precomputed gene":
        gene_input=GENE_OPTIONS[selected_gene]
    if gene_input and st.button("Analyze: Guide ranking",key="tab2_btn"):
        xgb_model=load_xgb()
        if xgb_model is None:
            st.error("xgboost pickle not found at Notebook/xgb_model.pkl")
            st.stop()
        target_gene=gene_input.strip().upper()
        #workflow status
        st.markdown("---")
        w1,w2,w3=st.columns(3)
        with w1:
            st.markdown("""
            <div class="phase-card phase-pending">
                <h4>Problem 1</h4>
                <h2>Mutation Detection</h2>
                <p>handled upstream by clinician</p>
            </div>""",unsafe_allow_html=True)
        with w2:
            st.markdown(f"""
            <div class="phase-card phase-pending">
                <h4>Problem 2</h4>
                <h2>{target_gene}</h2>
                <p>user specified, skipping gene selection</p>
            </div>""",unsafe_allow_html=True)
        with w3:
            st.markdown(f"""
            <div class="phase-card phase-active">
                <h4>Problem 3: Running</h4>
                <h2>Guide Design</h2>
                <p>PAM scan + xgboost + rag</p>
            </div>""",unsafe_allow_html=True)
        with st.spinner(f"fetching {target_gene} mRNA from ncbi..."):
            sequence,seq_id=fetch_ncbi_sequence(target_gene)
        if sequence is None:
            st.error(f"could not fetch sequence for {target_gene}. check spelling or try refseq ID (e.g. NM_000546 for TP53)")
            st.stop()
        st.success(f"sequence ready: {seq_id} ({len(sequence):,} bp)")
        with st.spinner(f"scanning {len(sequence):,} bp for PAM sites & scoring with xgboost..."):
            candidates=scan_pam(sequence,target_gene)
            scored=score_with_xgb(candidates,xgb_model)
        if not scored:
            st.warning("no guide RNAs passed quality filter (xgb_score >= 0.40)")
            st.stop()
        st.info(f"found {len(candidates)} PAM sites, {len(scored)} passed quality filter")
        st.markdown(f"### Top Guide RNAs for {target_gene}")
        st.caption("xgboost scores (0-1) trained on 5310 doench 2016 experimental measurements")
        for i,g in enumerate(scored[:5],1):
            with st.expander(f"rank {i} | xgb score: {g['xgb_score']} | {g['guide_sequence']}",expanded=(i==1)):
                st.markdown(f"""
                <div class="guide-card">
                    <span class="guide-seq">{g['guide_sequence']}</span>
                    <span class="score-badge">XGB: {g['xgb_score']}</span>
                </div>""",unsafe_allow_html=True)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("xgb score",g['xgb_score'])
                m2.metric("PAM",g['pam'])
                m3.metric("position",f"{g['position']} bp")
                m4.metric("position %",f"{g['position_pct']}%")
                st.code(f"30mer context: {g['seq30']}",language=None)

        #rag explanation:ml+literature
        st.markdown("---")
        st.markdown("### RAG + LLM Explanation")
        with st.spinner("fetching pubmed abstracts & generating explanation..."):
            try:
                t0=time.time()
                query_disease=disease_context.strip() if disease_context.strip() else target_gene
                papers=fetch_pubmed(query_disease,target_gene)
                ml_block=f"[ML PIPELINE] xgboost-ranked guide RNAs for {target_gene}:\n"
                for i,g in enumerate(scored[:5],1):
                    ml_block+=(f"rank {i}: {g['guide_sequence']} | xgb_score: {g['xgb_score']}"f" | PAM:{g['pam']} | position:{g['position']} bp\n")
                lit_block="[LITERATURE] relevant pubmed abstracts:\n"
                for i,p in enumerate(papers[:4],1):
                    lit_block+=f"[{i}] {p[:400]}...\n\n"
                question=(f"explain why rank 1 ({scored[0]['guide_sequence']},"
                            f"xgb_score {scored[0]['xgb_score']}) is the best crispr guide RNA"
                            f"candidate for {target_gene}. reference the literature")
                prompt=f"{ml_block}\n{lit_block}\nquestion:\n{question}\n\nanswer:"
                explanation=call_llm(prompt)
                latency=int((time.time()-t0)*1000)
                st.success("explanation generated")
                col1,col2=st.columns(2)
                col1.metric("response time",f"{latency} ms")
                col2.metric("pubmed abstracts",len(papers))
                st.subheader("AI generated explanation")
                st.markdown(explanation)
            except RuntimeError as e:
                st.warning(f"llm explanation unavailable:{e}")
#footer
st.markdown("---")
with st.expander("xgboost scoring reference (doench 2016, 93 features)"):
    st.markdown("""
| feature group | count | what it captures |
|---|---|---|
| GC content (30mer, 20mer, seed) | 3 | binding stability, seed region specificity |
| base frequencies (A/T/G/C) | 4 | nucleotide composition of guide |
| poly-T flag | 1 | 4+ consecutive T's terminate transcription |
| GC clamp | 1 | G/C at final position improves cas9 binding |
| homopolymer length | 1 | longer runs increase off-target risk |
| dinucleotide repeats | 1 | repetitive sequence patterns |
| unique dinucleotides | 1 | low diversity = higher off-target risk |
| seed region unique bases | 1 | seed specificity (last 12nt before PAM) |
| positional one-hot encoding | 80 | base identity at each of 20 guide positions |
| **total** | **93** | r2=0.27, comparable to azimuth sequence-only |
    """)
st.markdown("---")
st.caption("computational guidance only. all guides must be validated experimentally before clinical use")

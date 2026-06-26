"""
build_site.py  --  Live weekly tracker for the NFCI-gated leveraged-Nasdaq
regime strategy. Self-contained: fetches everything fresh from Yahoo + FRED
(no local cache/data-lake dependency) so it runs in GitHub Actions.

Rules (weekly):
  CASH      if NFCI 5y-percentile > 0.80 (armed 63 trading days)
  MODERATE  else if NDX < rising 200d SMA           -> 75% EQQQ + 25% MMF
  AGGRESSIVE else                                    -> 100% LQQ (2x net)

Writes site/index.html (Chart.js via CDN). Run: python3 tools/bravos-board/build_site.py
"""
from __future__ import annotations
import io, json, os, time, urllib.request, datetime as dt
import numpy as np, pandas as pd

TD = 252; COST = 7e-4
HERE = os.path.dirname(os.path.abspath(__file__))
SITE = HERE   # write index.html next to build_site.py (repo root on deploy)

# ----------------------------------------------------------------- fetch
def yf_close(ticker, start="1985-01-01"):
    import yfinance as yf
    for a in range(3):
        try:
            d = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if d is not None and len(d):
                s = d["Close"]; s = s.iloc[:,0] if isinstance(s, pd.DataFrame) else s
                return s.dropna()
        except Exception as e:
            print(f"  yf {ticker} retry {a}: {e}"); time.sleep(3)
    raise RuntimeError(f"yfinance failed: {ticker}")

def fred(sid):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1900-01-01"
    for a in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                df = pd.read_csv(io.StringIO(r.read().decode()))
            df.columns = ["date","val"]; df["date"]=pd.to_datetime(df["date"])
            df["val"]=pd.to_numeric(df["val"],errors="coerce")
            return df.dropna().set_index("date")["val"]
        except Exception as e:
            print(f"  fred {sid} retry {a}: {e}"); time.sleep(3)
    raise RuntimeError(f"FRED failed: {sid}")

def simulate_letf(ndx, irx, lev, er=0.0095, spread=0.0010):
    r = ndx.pct_change(); fin = irx.reindex(r.index).ffill().bfill().fillna(0.02)
    rl = lev*r - (lev-1)*fin/TD - (er+spread)/TD
    return (1+rl.fillna(0)).cumprod()

print("fetching ...")
ndx = yf_close("^NDX", "1985-01-01"); ndx.name="ndx"
irx = (yf_close("^IRX","1985-01-01")/100.0).reindex(ndx.index).ffill().bfill().clip(lower=0)
qqq = yf_close("QQQ","1999-01-01"); spy = yf_close("SPY","1993-01-01")
nfci = fred("NFCI")
letf2 = simulate_letf(ndx, irx, 2.0); letf3 = simulate_letf(ndx, irx, 3.0)
cash = (1+irx/TD).cumprod()

df = pd.DataFrame({"ndx":ndx,"letf2":letf2,"letf3":letf3,"cash":cash})
df["qqq"]=qqq.reindex(df.index); df["spy"]=spy.reindex(df.index)
df["nfci"]=nfci.reindex(df.index, method="ffill")
df = df[df.index>="1998-01-01"].copy()
df[["qqq","spy"]] = df[["qqq","spy"]].ffill()

# ----------------------------------------------------------------- signal
sma200 = df["ndx"].rolling(200).mean()
trend_on = (df["ndx"]>sma200) & (sma200>sma200.shift(21))
nfci_pct = df["nfci"].rolling(1260, min_periods=252).rank(pct=True)
tight = (nfci_pct>0.80).rolling(63, min_periods=1).max().astype(bool)
state = pd.Series("AGG", index=df.index)
state[~trend_on] = "MOD"
state[tight] = "CASH"

# weekly cadence + position effective next day
idx = df.index; wk = pd.Series(idx).dt.isocalendar(); key=(wk.year*100+wk.week).values
wmask = np.concatenate([[True], key[1:]!=key[:-1]])
ar = np.arange(len(df)); pos=np.where(wmask,ar,-1); last=np.maximum.accumulate(pos); last[last<0]=0
codes = state.map({"AGG":0,"MOD":1,"CASH":2}).values
held = codes[last]; he = np.empty_like(held); he[0]=2; he[1:]=held[:-1]

def equity(levcol):
    W = np.array([[1.,0,0],[0,.75,.25],[0,0,1.]])
    R = np.column_stack([df[levcol].pct_change().fillna(0).values,
                         df["qqq"].pct_change().fillna(0).values,
                         df["cash"].pct_change().fillna(0).values])
    w = W[he]; port=(w*R).sum(1)
    chg=np.empty(len(he),bool); chg[0]=False; chg[1:]=he[1:]!=he[:-1]
    turn=np.zeros(len(he)); turn[1:]=np.abs(w[1:]-w[:-1]).sum(1)
    return pd.Series(np.cumprod(1+port-np.where(chg,COST*turn,0.)), index=idx)

start = "2000-01-01"
eq2 = equity("letf2"); eq3 = equity("letf3")
sub = df.index>=pd.Timestamp(start)
def norm(s): s=s[sub]; return s/s.iloc[0]
strat=norm(eq2); strat3=norm(eq3)
spyN=norm(df["spy"]); qqqN=norm(df["qqq"])

# ----------------------------------------------------------------- metrics
def cagr(s): yrs=(s.index[-1]-s.index[0]).days/365.25; return (s.iloc[-1]/s.iloc[0])**(1/yrs)-1
def mdd(s): return (s/s.cummax()-1).min()
def sharpe(s):
    r=s.pct_change().dropna(); rc=df["cash"].pct_change().reindex(r.index).fillna(0)
    ex=r-rc; return np.sqrt(TD)*ex.mean()/ex.std() if ex.std()>0 else 0
def ytd(s):
    y=s[s.index>=f"{idx[-1].year}-01-01"]; return y.iloc[-1]/y.iloc[0]-1 if len(y)>1 else 0
def mar(s): d=mdd(s); return cagr(s)/abs(d) if d<0 else 0
M = {
  "CAGR": cagr(strat), "MaxDD": mdd(strat), "MAR": mar(strat), "Sharpe": sharpe(strat),
  "YTD": ytd(strat), "mult": strat.iloc[-1], "xSPY": strat.iloc[-1]/spyN.iloc[-1],
  "xQQQ": strat.iloc[-1]/qqqN.iloc[-1],
  "spy_cagr": cagr(spyN), "qqq_cagr": cagr(qqqN), "spy_mdd": mdd(spyN),
  "cagr3": cagr(strat3), "mdd3": mdd(strat3), "mult3": strat3.iloc[-1],
  "pcash": (pd.Series(he,index=idx)[sub]==2).mean(),
  "pagg": (pd.Series(he,index=idx)[sub]==0).mean(),
  "pmod": (pd.Series(he,index=idx)[sub]==1).mean(),
}

# ----------------------------------------------------------------- current status + gate proximity
asof = idx[-1]
cur = {0:"AGGRESSIVE",1:"MODERATE",2:"CASH"}[int(he[-1])]
ndx_now=float(df["ndx"].iloc[-1]); sma_now=float(sma200.iloc[-1])
sma_rising=bool(sma200.iloc[-1]>sma200.iloc[-22])
pct_above = ndx_now/sma_now-1
nfci_now=float(df["nfci"].iloc[-1]); nfci_pct_now=float(nfci_pct.iloc[-1])
# NFCI level that maps to the 0.80 trigger percentile, over the last 5y
nfci_win = df["nfci"].iloc[-1260:]
nfci_trigger_level = float(nfci_win.quantile(0.80))
ACTION = {
  "AGGRESSIVE": "Hold 100% LQQ (2x Nasdaq). Risk on.",
  "MODERATE":   "Hold 75% EQQQ + 25% money-market fund. Trend soft, conditions calm.",
  "CASH":       "Hold 100% money-market fund. Financial conditions tight.",
}[cur]

# next NFCI release = next Wednesday (Chicago Fed publishes Wed ~8:30 ET for prior week).
# weekly signal rolls at the start of each ISO week (Monday); the page holds that
# weekly signal steady and only refreshes prices/gate-meters daily.
today = dt.date.today()
next_nfci = today + dt.timedelta(days=(2 - today.weekday()) % 7)   # next Wed (incl. today)
next_eval = today + dt.timedelta(days=(0 - today.weekday()) % 7)   # next Mon (incl. today)

# ----------------------------------------------------------------- signal history (weekly)
wsel = idx[wmask]
hist = pd.DataFrame({"date":idx,"state":pd.Series(he,index=idx).map({0:"AGG",1:"MOD",2:"CASH"}),
                     "ndx":df["ndx"].values,"nfci":df["nfci"].values}).set_index("date")
histw = hist.loc[wsel].copy()
# keep only rows where state changed, plus the latest 12 weeks
changes = histw[histw["state"]!=histw["state"].shift()]
recent_changes = changes.tail(12)

# ----------------------------------------------------------------- chart data (weekly downsample)
chart_idx = strat.index[strat.index.isin(wsel)]
def series(s):
    v=s.reindex(chart_idx); return [round(float(x),3) for x in v.values]
DATA = {
  "dates":[d.strftime("%Y-%m-%d") for d in chart_idx],
  "strat":series(strat), "strat3":series(strat3), "spy":series(spyN), "qqq":series(qqqN),
}
# recent Nasdaq-100 + 200d SMA (3y daily) and NFCI (3y weekly) for the small charts
nxi = df.index[-756:]
DATA["nx_dates"] = [d.strftime("%Y-%m-%d") for d in nxi]
DATA["nx_ndx"]   = [round(float(x)) for x in df["ndx"].reindex(nxi).values]
_smar = sma200.reindex(nxi).values
DATA["nx_sma"]   = [None if pd.isna(x) else round(float(x)) for x in _smar]
nfw = df["nfci"].reindex(wsel).dropna().iloc[-160:]
DATA["nf_dates"] = [d.strftime("%Y-%m-%d") for d in nfw.index]
DATA["nf_val"]   = [round(float(x),2) for x in nfw.values]
DATA["nf_trig"]  = round(float(nfci_trigger_level),2)

# ----------------------------------------------------------------- render
def pc(x,d=1): return f"{x*100:.{d}f}%"
def signl(x): return ("+" if x>=0 else "")+pc(x)
badge_color={"AGGRESSIVE":"#E3120B","MODERATE":"#C77B0A","CASH":"#006BA2"}[cur]

rows_html=""
for d,r in recent_changes[::-1].iterrows():
    col={"AGG":"#E3120B","MOD":"#C77B0A","CASH":"#006BA2"}[r["state"]]
    nm={"AGG":"Aggressive","MOD":"Moderate","CASH":"Cash"}[r["state"]]
    rows_html+=f"<tr><td>{d.strftime('%d %b %Y')}</td><td style='color:{col};font-weight:700'>{nm}</td><td>{r['ndx']:,.0f}</td><td>{r['nfci']:+.2f}</td></tr>"

# trend cushion: how far NDX can fall before crossing SMA200
trend_drop = pct_above
trend_bar = max(0,min(100, (pct_above+0.10)/0.30*100))   # -10%..+20% mapped to 0..100
nfci_bar = max(0,min(100, nfci_pct_now*100))

# marker positions + plain-English gate sentences
trend_pos = max(2, min(98, (pct_above+0.10)/0.30*100))   # scale -10%..+20%
nfci_pos  = max(2, min(98, nfci_pct_now*100))
sma_word = "rising" if sma_rising else "falling"
if pct_above >= 0:
    trend_line = (f"The Nasdaq-100 ({ndx_now:,.0f}) is {signl(pct_above)} above its 200-day "
                  f"({sma_now:,.0f}, {sma_word}). It would have to fall {pc(abs(pct_above))} to drop to Moderate.")
else:
    trend_line = (f"The Nasdaq-100 ({ndx_now:,.0f}) is {signl(pct_above)} vs its 200-day "
                  f"({sma_now:,.0f}, {sma_word}) &mdash; below the line, so the trend gate says Moderate.")
if nfci_pct_now < 0.80:
    nfci_line = (f"NFCI is {nfci_now:+.2f}, at the {pc(nfci_pct_now,0)} mark of its 5-year range (loose). "
                 f"Cash needs the top 20% (NFCI above {nfci_trigger_level:+.2f}) &mdash; well above here.")
else:
    nfci_line = (f"NFCI is {nfci_now:+.2f}, in the top 20% of its 5-year range &mdash; conditions tight, so the gate says Cash.")

# ---- alerts: email on signal change, or when a gate newly gets close --------
import smtplib, ssl
from email.mime.text import MIMEText
STATE = os.path.join(HERE, "state.json")
prev = {}
if os.path.exists(STATE):
    try: prev = json.load(open(STATE))
    except Exception: prev = {}
trend_near = (0 <= pct_above < 0.04)        # within 4% above the 200-day
cash_near  = (nfci_pct_now > 0.70)          # within 10 percentile pts of the 0.80 cash trigger
alerts = []
if prev:  # don't alert on the very first run (no baseline)
    if prev.get("signal") != cur:
        alerts.append(f"SIGNAL CHANGED: {prev.get('signal')} -> {cur}. {ACTION}")
    if trend_near and not prev.get("trend_near"):
        alerts.append(f"Trend gate approaching: Nasdaq-100 only {signl(pct_above)} above its 200-day. A fall below it -> Moderate.")
    if cash_near and not prev.get("cash_near"):
        alerts.append(f"Conditions tightening: NFCI at {pc(nfci_pct_now,0)} of its 5y range (cash at 80%). NFCI {nfci_now:+.2f} vs trigger {nfci_trigger_level:+.2f}.")
json.dump({"signal":cur,"trend_near":bool(trend_near),"cash_near":bool(cash_near),
           "asof":str(asof.date()),"updated":dt.datetime.now(dt.timezone.utc).isoformat()},
          open(STATE,"w"), indent=2)
gm_user=os.environ.get("GMAIL_USER"); gm_pw=os.environ.get("GMAIL_APP_PW")
to=os.environ.get("ALERT_TO", gm_user or "")
if alerts and gm_user and gm_pw:
    chg = any("CHANGED" in a for a in alerts)
    body=("The Regime — alert\n\n"+"\n\n".join(alerts)+
          f"\n\nCurrent signal: {cur}\nLive: https://soylee22.github.io/the-regime/\nas of {asof.date()}")
    m=MIMEText(body); m["Subject"]=f"[The Regime] {cur}"+(" — SIGNAL CHANGE" if chg else " — gate watch")
    m["From"]=gm_user; m["To"]=to
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com",465,context=ssl.create_default_context()) as s:
            s.login(gm_user, gm_pw); s.sendmail(gm_user,[to],m.as_string())
        print(f"alert email sent ({len(alerts)})")
    except Exception as e:
        print("email failed:", e)
elif alerts:
    print("alerts (no email creds set):", alerts)

tmpl = open(os.path.join(HERE,"template.html")).read()
repl = {
 "@@TRENDPOS@@": f"{trend_pos:.0f}", "@@NFCIPOS@@": f"{nfci_pos:.0f}",
 "@@TRENDLINE@@": trend_line, "@@NFCILINE@@": nfci_line,
 "@@ASOF@@": asof.strftime("%d %b %Y"),
 "@@CUR@@": cur, "@@BADGE@@": badge_color, "@@ACTION@@": ACTION,
 "@@CAGR@@": pc(M["CAGR"]), "@@MAXDD@@": pc(M["MaxDD"]), "@@MAR@@": f"{M['MAR']:.2f}",
 "@@SHARPE@@": f"{M['Sharpe']:.2f}", "@@YTD@@": signl(M["YTD"]), "@@MULT@@": f"{M['mult']:.0f}x",
 "@@XSPY@@": f"{M['xSPY']:.1f}x", "@@XQQQ@@": f"{M['xQQQ']:.1f}x",
 "@@SPYCAGR@@": pc(M["spy_cagr"]), "@@QQQCAGR@@": pc(M["qqq_cagr"]),
 "@@PAGG@@": pc(M["pagg"],0), "@@PMOD@@": pc(M["pmod"],0), "@@PCASH@@": pc(M["pcash"],0),
 "@@CAGR3@@": pc(M["cagr3"]), "@@MDD3@@": pc(M["mdd3"]),
 "@@NDXNOW@@": f"{ndx_now:,.0f}", "@@SMANOW@@": f"{sma_now:,.0f}",
 "@@PCTABOVE@@": signl(pct_above), "@@SMARISE@@": "rising" if sma_rising else "falling",
 "@@TRENDBAR@@": f"{trend_bar:.0f}", "@@TRENDSTATE@@": "above" if pct_above>0 else "below",
 "@@NFCINOW@@": f"{nfci_now:+.2f}", "@@NFCIPCT@@": pc(nfci_pct_now,0),
 "@@NFCITRIG@@": f"{nfci_trigger_level:+.2f}", "@@NFCIBAR@@": f"{nfci_bar:.0f}",
 "@@NEXTNFCI@@": next_nfci.strftime("%a %d %b %Y"), "@@NEXTEVAL@@": next_eval.strftime("%a %d %b %Y"),
 "@@ROWS@@": rows_html, "@@DATA@@": json.dumps(DATA),
 "@@BUILT@@": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
}
for k,v in repl.items(): tmpl=tmpl.replace(k,str(v))
open(os.path.join(SITE,"index.html"),"w").write(tmpl)
print(f"wrote {SITE}/index.html")
print(f"signal={cur} | CAGR {pc(M['CAGR'])} MaxDD {pc(M['MaxDD'])} xSPY {M['xSPY']:.1f} | NFCI {nfci_now:+.2f} ({pc(nfci_pct_now,0)}) trig>{nfci_trigger_level:+.2f}")
print(f"wrote {SITE}/index.html")

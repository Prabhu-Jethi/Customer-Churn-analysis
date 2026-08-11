const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

function scrollToSection(id){
  document.getElementById(id)?.scrollIntoView({behavior:"smooth",block:"start"});
}
$$("[data-scroll]").forEach(el => el.addEventListener("click", () => scrollToSection(el.dataset.scroll)));

const workflowData=[
  ["01 · Customer data","Start with the customer story.","Profile, tenure, contract and service attributes become the input to the prediction layer."],
  ["02 · ML prediction","Turn customer attributes into risk.","The selected XGBoost model produces a probability score that can be consumed by the product layer."],
  ["03 · SHAP analysis","Explain why the score changed.","Feature-level contributions surface the strongest risk drivers instead of leaving the prediction as a black box."],
  ["04 · Retention action","Convert risk into a next step.","Prioritize the customer and recommend an intervention based on the strongest signals."]
];
$$(".pipeline-step").forEach((el,i)=>el.addEventListener("click",()=>{
  $$(".pipeline-step").forEach(x=>x.classList.remove("active")); el.classList.add("active");
  const d=workflowData[i];
  $("#pipelineDetail").innerHTML=`<span>${d[0]}</span><strong>${d[1]}</strong><p>${d[2]}</p>`;
}));

const tenure=$("#tenure"),charges=$("#charges"),contract=$("#contract");
const defaultState={tenure:65,charges:29,contract:"monthly"};
let committed={...defaultState};

function score(t,c,ct){
  // Demo-only heuristic. Replace this function with the FastAPI response later.
  let r=0.22;
  r += Math.max(0,(48-t)/48)*0.25;
  r += Math.max(0,(c-55)/65)*0.12;
  r += ct==="monthly"?0.28:ct==="one"?0.10:-0.08;
  return Math.max(0.03,Math.min(0.97,r));
}
function riskClass(r){return r<0.40?"low":r<0.70?"medium":"high"}
function riskLabel(r){return r<0.40?"Low risk":r<0.70?"Medium risk":"High risk"}
function recommendation(r){
  if(r>=.70)return["Prioritize this customer.","Offer a contract incentive and technical support option to reduce churn risk."];
  if(r>=.40)return["Monitor this customer.","Consider proactive outreach and a service review before risk increases."];
  return["Maintain the relationship.","Customer signals are relatively stable. Continue engagement and monitor changes."];
}
function calculate(t,c,ct){
  const r=score(t,c,ct), rc=riskClass(r);
  const ci=ct==="monthly"?.28:ct==="one"?.12:.05;
  const ti=Math.max(.02,((48-t)/48)*.24);
  const ch=Math.max(.02,(Math.max(0,c-35)/85)*.18);
  const si=r>.70?.11:r>.40?.07:.03;
  return {r,rc,ci,ti,ch,si};
}
function applyPrediction(showToast=false){
  const t=Number(tenure.value),c=Number(charges.value),ct=contract.value;
  const d=calculate(t,c,ct),pct=(d.r*100).toFixed(1);
  committed={tenure:t,charges:c,contract:ct};
  const card=$(".risk-card"), num=$("#riskNumber"), badge=$("#riskLabel");
  card.classList.remove("risk-low","risk-medium","risk-high"); card.classList.add(`risk-${d.rc}`);
  num.classList.remove("risk-low","risk-medium","risk-high"); num.classList.add(`risk-${d.rc}`);
  badge.className=`risk-badge ${d.rc}`; badge.textContent=riskLabel(d.r); num.textContent=`${pct}%`;
  $("#tenureValue").textContent=`${t} mo`; $("#tenureMeta").textContent=t<18?"Short tenure":"Established customer";
  $("#chargesValue").textContent=`$${c}`; $("#chargesMeta").textContent=c>60?"Above median":"Moderate charges";
  $("#contractValue").textContent=ct==="monthly"?"Monthly":ct==="one"?"1 year":"2 year";
  $("#tenureOut").textContent=t; $("#chargesOut").textContent=c;
  $("#impactContract").textContent=`+${d.ci.toFixed(2)}`; $("#impactTenure").textContent=`+${d.ti.toFixed(2)}`; $("#impactCharges").textContent=`+${d.ch.toFixed(2)}`; $("#impactSupport").textContent=`+${d.si.toFixed(2)}`;
  $("#barContract").style.width=`${Math.max(12,d.ci/.30*100)}%`; $("#barTenure").style.width=`${Math.max(10,d.ti/.24*100)}%`; $("#barCharges").style.width=`${Math.max(10,d.ch/.18*100)}%`; $("#barSupport").style.width=`${Math.max(10,d.si/.12*100)}%`;
  const rec=recommendation(d.r); $("#recommendationTitle").textContent=rec[0]; $("#recommendationText").textContent=rec[1];
  $("#historyRisk").textContent=`${pct}% · ${riskLabel(d.r).replace(" risk","")}`;
  $("#explainContract").textContent=`+${d.ci.toFixed(2)}`; $("#explainTenure").textContent=`+${d.ti.toFixed(2)}`; $("#explainCharges").textContent=`+${d.ch.toFixed(2)}`;
  $("#explainText").textContent=d.rc==="high"?"Contract type is the strongest current driver, with shorter tenure and charges adding risk.":d.rc==="medium"?"Contract type is the leading signal; tenure and monthly charges are contributing moderate risk.":"Customer signals are comparatively stable. Contract, tenure and charges currently contribute limited risk.";
  $("#heroRisk").textContent=`${pct}%`; $("#heroRiskLabel").textContent=riskLabel(d.r);
  $("#heroRiskLabel").className=d.rc==="low"?"status-low":d.rc==="medium"?"status-medium":"status-high";
  if(showToast) toast("Prediction updated — demo model response applied.");
}
[tenure,charges,contract].forEach(el=>el.addEventListener("input",()=> {
  // Live preview: update the control values but keep the prediction state visually current.
  applyPrediction(false);
}));
$("#predictBtn").addEventListener("click",()=>applyPrediction(true));
$("#predictBtn2").addEventListener("click",()=>{scrollToSection("preview");applyPrediction(true);setTab("overview")});
$("#resetPrediction").addEventListener("click",()=>{tenure.value=defaultState.tenure;charges.value=defaultState.charges;contract.value=defaultState.contract;applyPrediction(true);});

function setTab(tab){
  $$(".app-side button").forEach(b=>b.classList.toggle("side-active",b.dataset.tab===tab));
  $$(".app-view").forEach(v=>v.classList.toggle("active",v.dataset.view===tab));
}
$$(".app-side button").forEach(b=>b.addEventListener("click",()=>setTab(b.dataset.tab)));

function toast(message){const el=$("#toast");el.textContent=message;el.classList.add("show");clearTimeout(window.__toast);window.__toast=setTimeout(()=>el.classList.remove("show"),2400)}
const modal=$("#modal");
function openModal(){const d=calculate(Number(tenure.value),Number(charges.value),contract.value),rec=recommendation(d.r);$("#modalTitle").textContent=rec[0];$("#modalBody").textContent=rec[1]+` Current estimated churn probability: ${(d.r*100).toFixed(1)}%.`;modal.classList.add("open")}
$("#recommendationBtn").addEventListener("click",openModal);$("#modalClose").addEventListener("click",()=>modal.classList.remove("open"));$("#modal").addEventListener("click",e=>{if(e.target===modal)modal.classList.remove("open")});$("#modalAction").addEventListener("click",()=>{modal.classList.remove("open");toast("Recommendation acknowledged.");});

$(".menu-btn").addEventListener("click",()=>document.body.classList.toggle("menu-open"));
$$(".nav-links a").forEach(a=>a.addEventListener("click",()=>document.body.classList.remove("menu-open")));

applyPrediction(false);

$("#exploreAnalytics")?.addEventListener("click",()=>{setTab("analytics");scrollToSection("preview");});

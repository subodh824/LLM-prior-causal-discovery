/* Shared helpers used by every report page.
   Load this file before any page script - the pages depend on it. */

   const LEGS = [
    "supplier_to_manufacturer",
    "manufacturer_to_distributor",
    "distributor_to_customer",
    "dataco"
  ];
  
  const BASE = "../experiments/";
  
  // Which leg is this page showing? Comes from the ?leg= query parameter.
  // Falls back to the distributor leg, the first one with full results.
  const leg = new URLSearchParams(location.search).get("leg") || LEGS[2];
  const legDir = BASE + leg + "/";
  const legTitle = leg.replaceAll("_", " ");
  
  // Everything we know about each prior type, in one place:
  // where it sorts in tables, which row colour it gets (a CSS class),
  // which colour its chart line would use, and how to label it.
  // Adding a new LLM provider = adding one line here.
  const PRIOR_TYPES = {
    baseline:         { order: 0, css: "",        color: "#555555", label: "Baseline" },
    random:           { order: 1, css: "random",  color: "#d9962e", label: "Random" },
    gemini:           { order: 2, css: "gemini",  color: "#3a8a3a", label: "Gemini draws" },
    gemini_consensus: { order: 3, css: "gemini",  color: "#2f6f2f", label: "Gemini consensus" },
    groq:             { order: 4, css: "groq",    color: "#2e6fd9", label: "Groq draws" },
    groq_consensus:   { order: 5, css: "groq",    color: "#24589f", label: "Groq consensus" },
    perfect:          { order: 6, css: "perfect", color: "#9a9ab5", label: "Perfect (oracle)" },
  };
  
  // Look a type up regardless of capitalisation ("Baseline" vs "baseline").
  // Unknown types get a sensible fallback instead of crashing the page.
  function typeInfo(type) {
    const key = String(type || "").toLowerCase();
    if (PRIOR_TYPES[key]) return PRIOR_TYPES[key];
    return { order: 9, css: "", color: "#888888", label: type };
  }
  
  // Format a value as "mean ± ci". When there is no CI, show just the
  // mean. When the value itself is missing, show a dash.
  function fmt(mean, ci) {
    if (mean == null) return "—";
    if (ci == null) return Number(mean).toFixed(2);
    return Number(mean).toFixed(2) + " ± " + Number(ci).toFixed(2);
  }
  
  // Fetch a JSON file. Returns null when the file does not exist yet,
  // so pages can show a "no results yet" note instead of breaking.
  async function getJSON(path) {
    try {
      const response = await fetch(path);
      if (!response.ok) return null;
      return await response.json();
    } catch (err) {
      return null;
    }
  }
  
  // Navigation bar shown at the top of every page.
  function navBar() {
    const parts = ['<a href="index.html">home</a>'];
    for (const name of LEGS) {
      const short = name.split("_")[0];
      parts.push(
        `<b>${short}:</b>` +
        ` <a href="priors.html?leg=${name}">priors</a>` +
        ` <a href="discovery.html?leg=${name}">discovery</a>` +
        ` <a href="improvement.html?leg=${name}">improvement</a>` +
        ` <a href="dags.html?leg=${name}">dags</a>` +
        ` <a href="validation.html?leg=${name}">validation</a>` +
        ` <a href="figures.html?leg=${name}">figures</a>` +
        ` <a href="attribution.html?leg=${name}">attribution</a>` +
        ` <a href="intervention.html?leg=${name}">intervention</a>`
      );
    }
    return "<nav>" + parts.join(" | ") + "</nav>";
  }
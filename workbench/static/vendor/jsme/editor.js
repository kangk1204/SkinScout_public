// Bridge between the vendored JSME applet and the Workbench page.
// Kept external so this frame needs no script hash of its own; the inline
// scripts that still require 'unsafe-inline' here are GWT's own bootstrap.
var applet = null;

function jsmeOnLoad() {
  applet = new JSApplet.JSME("jsme", "100%", "340px", { options: "newlook,star" });
  applet.setCallBack("AfterStructureModified", publish);
  publish();
}

function publish() {
  var value = applet ? applet.smiles() : "";
  document.getElementById("smiles").textContent = value || "(빈 구조)";
  return value;
}

document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("use").addEventListener("click", function () {
    var value = publish();
    if (!value) return;
    parent.postMessage(
      { source: "skinscout-sketcher", smiles: value },
      window.location.origin
    );
  });
  document.getElementById("clear").addEventListener("click", function () {
    if (applet) { applet.reset(); publish(); }
  });
});

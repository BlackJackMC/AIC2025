window.addEventListener("load", () => {
  const params = new URLSearchParams(window.location.search);
  const frameId = params.get("frame");

  if (frameId) {
    const el = document.getElementById("frame-" + frameId);
    if (el) {
      el.classList.add("highlight");
      // Scroll smoothly AFTER all images are fully loaded
      setTimeout(() => {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 100);
    }
  }
});

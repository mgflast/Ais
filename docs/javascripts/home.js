// Tag the landing page with a `home` body class so its content can be narrowed and
// the auto-injected site-name heading hidden. `hide: navigation` still leaves the
// sidebar element in the DOM, so detect the home by comparing the current path to the
// header logo's link (which always points at the site root). Runs on load and on
// every instant-navigation change.
(function () {
  function isHome() {
    var logo = document.querySelector(".md-header__button.md-logo");
    if (!logo) return false;
    var root = new URL(logo.getAttribute("href"), window.location.href).pathname.replace(/index\.html$/, "");
    var here = window.location.pathname.replace(/index\.html$/, "");
    return here === root;
  }
  function update() {
    document.body.classList.toggle("home", isHome());
  }
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(update);
  } else {
    document.addEventListener("DOMContentLoaded", update);
  }
})();

/* Scene player for the explanatory animations on the CoPE project page.
 *
 * Each `figure.anim` declares `data-durations="ms,ms,..."`, one entry per
 * scene. The player sets `data-scene="N"` on the figure; all drawing is done
 * by CSS keyed on that attribute. Captions (`.anim-cap[data-for="N"]`) and
 * scene dots follow the same index.
 *
 * `createPlayer` is pure (no DOM) so the timing can be unit-tested; see
 * tests/anim.test.js. The DOM wiring at the bottom runs only in a browser.
 */
(function (root) {
  'use strict';

  function createPlayer(opts) {
    var durations = opts.durations;
    var onScene = opts.onScene;
    var timer = opts.timer || {
      set: function (fn, ms) { return setTimeout(fn, ms); },
      clear: function (id) { clearTimeout(id); }
    };
    var reduced = !!opts.reduced;
    var last = durations.length - 1;
    var current = reduced ? last : 0;
    var playing = false;
    var pending = null;

    function clearPending() {
      if (pending !== null) {
        timer.clear(pending);
        pending = null;
      }
    }

    function schedule() {
      clearPending();
      if (!playing || reduced) return;
      pending = timer.set(function () {
        pending = null;
        go(current === last ? 0 : current + 1);
      }, durations[current]);
    }

    function go(i) {
      current = i;
      onScene(i);
      schedule();
    }

    function play() {
      if (playing) return;
      playing = true;
      go(current);
    }

    function pause() {
      playing = false;
      clearPending();
    }

    function replay() {
      playing = true;
      go(0);
    }

    return {
      play: play,
      pause: pause,
      replay: replay,
      go: go,
      get scene() { return current; },
      get playing() { return playing; }
    };
  }

  /* ---------- DOM wiring ---------- */

  function mount(fig) {
    var durations = (fig.getAttribute('data-durations') || '')
      .split(',')
      .map(function (s) { return parseInt(s, 10); })
      .filter(function (n) { return n > 0; });
    if (!durations.length) return;

    var caps = fig.querySelectorAll('.anim-cap');
    var dotsBox = fig.querySelector('.anim-dots');
    var dots = [];

    if (dotsBox) {
      durations.forEach(function (_, i) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'anim-dot';
        b.setAttribute('aria-label', 'Scene ' + (i + 1) + ' of ' + durations.length);
        b.addEventListener('click', function () { player.go(i); });
        dotsBox.appendChild(b);
        dots.push(b);
      });
    }

    function render(i) {
      // Clearing and re-setting the attribute restarts the scene's CSS
      // animations even when the index is unchanged (replay, resume).
      fig.removeAttribute('data-scene');
      void fig.offsetWidth;
      fig.setAttribute('data-scene', String(i));
      for (var c = 0; c < caps.length; c++) {
        caps[c].classList.toggle('is-on', caps[c].getAttribute('data-for') === String(i));
      }
      dots.forEach(function (d, k) {
        d.classList.toggle('is-on', k === i);
        if (k === i) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
      });
    }

    var reduced = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var player = createPlayer({ durations: durations, onScene: render, reduced: reduced });

    var replay = fig.querySelector('.anim-replay');
    if (replay) replay.addEventListener('click', function () { player.replay(); });

    if ('IntersectionObserver' in root) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) player.play(); else player.pause();
        });
      }, { threshold: 0.35 });
      io.observe(fig);
    } else {
      player.play();
    }

    if (reduced) render(player.scene);
  }

  // Results charts: bars grow and lines draw once, when first scrolled into view.
  function mountCharts() {
    var charts = document.querySelectorAll('.chart');
    if (!charts.length) return;
    if (!('IntersectionObserver' in root)) {
      for (var j = 0; j < charts.length; j++) charts[j].classList.add('is-in');
      return;
    }
    var cio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          cio.unobserve(e.target);
        }
      });
    }, { threshold: 0.2 });
    for (var i = 0; i < charts.length; i++) cio.observe(charts[i]);
  }

  function init() {
    var figs = document.querySelectorAll('figure.anim');
    for (var i = 0; i < figs.length; i++) mount(figs[i]);
    mountCharts();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createPlayer: createPlayer };
  }
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
  }
})(typeof window !== 'undefined' ? window : {});

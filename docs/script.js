/* ii — landing page interactions */
(function () {
  'use strict';

  /* ---- Navbar scroll state ---- */
  const navbar = document.getElementById('navbar');
  const onScroll = () => navbar.classList.toggle('scrolled', window.scrollY > 20);
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  /* ---- Mobile menu ---- */
  const hamburger = document.getElementById('hamburger');
  const navMenu = document.getElementById('navMenu');
  hamburger.addEventListener('click', () => {
    const open = navMenu.classList.toggle('open');
    hamburger.classList.toggle('open', open);
    hamburger.setAttribute('aria-expanded', String(open));
  });
  navMenu.querySelectorAll('a').forEach(a =>
    a.addEventListener('click', () => {
      navMenu.classList.remove('open');
      hamburger.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
    })
  );

  /* ---- Quick-start tabs ---- */
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const id = tab.dataset.tab;
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(id).classList.add('active');
    });
  });

  /* ---- Copy buttons ---- */
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const target = document.getElementById(btn.dataset.copy);
      if (!target) return;
      const text = target.innerText.replace(/^\$\s/gm, '').trim();
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        const r = document.createRange(); r.selectNode(target);
        const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
        try { document.execCommand('copy'); } catch (_) {}
        sel.removeAllRanges();
      }
      const icon = btn.querySelector('i');
      const prev = icon.className;
      icon.className = 'fas fa-check';
      btn.classList.add('copied');
      setTimeout(() => { icon.className = prev; btn.classList.remove('copied'); }, 1500);
    });
  });

  /* ---- Scroll reveal ---- */
  const revealTargets = document.querySelectorAll(
    '.card, .backend-card, .engine-text, .engine-code, .section-head, .highlights-grid, .cta-inner, .tabs'
  );
  revealTargets.forEach(el => el.classList.add('reveal'));
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { threshold: 0.12 });
    revealTargets.forEach(el => io.observe(el));
  } else {
    revealTargets.forEach(el => el.classList.add('in'));
  }

  /* ---- Populate downloads from GitHub Releases ---- */
  (function downloads() {
    const cards = {
      linux:   document.getElementById('dl-linux'),
      windows: document.getElementById('dl-windows'),
      macos:   document.getElementById('dl-macos')
    };
    if (!cards.linux) return;

    const fmtSize = (b) => b ? (b / 1048576).toFixed(1) + ' MB' : '';
    const pickAsset = (assets, os) => {
      const m = assets.filter(a => /\.zip$/i.test(a.name) && a.name.toLowerCase().includes('-' + os + '-'));
      if (!m.length) return null;
      return m.find(a => /x64|x86_64|amd64/i.test(a.name)) || m[0];
    };

    fetch('https://api.github.com/repos/DEgITx/ii/releases/latest', {
      headers: { Accept: 'application/vnd.github+json' }
    })
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(rel => {
        const tag = rel.tag_name || '';
        const vBadge = document.getElementById('dl-version');
        if (vBadge && tag) vBadge.textContent = tag;
        const assets = rel.assets || [];
        Object.keys(cards).forEach(os => {
          const card = cards[os];
          const asset = pickAsset(assets, os);
          if (asset) {
            card.href = asset.browser_download_url;
            card.removeAttribute('target');           // direct download, same tab
            const meta = card.querySelector('[data-meta]');
            const arch = /arm64|aarch64/i.test(asset.name) ? 'arm64' : 'x86-64';
            if (meta) meta.textContent = arch + ' · ' + fmtSize(asset.size);
          }
          // if no matching asset, keep the default link to releases/latest
        });
      })
      .catch(() => { /* keep fallback links to releases page */ });
  })();

  /* ---- Current year ---- */
  const y = document.getElementById('year');
  if (y) y.textContent = new Date().getFullYear();

  /* ---- Neural network hero canvas ---- */
  const canvas = document.getElementById('neural');
  if (canvas && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const ctx = canvas.getContext('2d');
    let w, h, dpr, nodes, raf;
    const COLORS = ['#2DE2C8', '#6366F1', '#A855F7'];

    function size() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      w = rect.width; h = rect.height;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      build();
    }

    function build() {
      const count = Math.round(Math.min(70, Math.max(28, (w * h) / 22000)));
      nodes = [];
      for (let i = 0; i < count; i++) {
        nodes.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.28,
          vy: (Math.random() - 0.5) * 0.28,
          r: Math.random() * 1.8 + 1.2,
          c: COLORS[i % COLORS.length]
        });
      }
    }

    function frame() {
      ctx.clearRect(0, 0, w, h);
      const LINK = 140;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        a.x += a.vx; a.y += a.vy;
        if (a.x < 0 || a.x > w) a.vx *= -1;
        if (a.y < 0 || a.y > h) a.vy *= -1;
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d = Math.hypot(dx, dy);
          if (d < LINK) {
            ctx.globalAlpha = (1 - d / LINK) * 0.32;
            ctx.strokeStyle = a.c;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      }
      for (const n of nodes) {
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = n.c;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(frame);
    }

    size();
    frame();
    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(size, 200); });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) cancelAnimationFrame(raf);
      else frame();
    });
  }
})();

(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Animated Gaussian wordmark. Soft, rotating planar splats are clipped to
  // the letterforms, echoing FlaRe's representation without a bitmap asset.
  function initWordmark() {
    const canvas = document.querySelector('#flare-wordmark');
    if (!canvas || !canvas.getContext) {
      document.documentElement.classList.add('no-canvas');
      return;
    }
    const ctx = canvas.getContext('2d');
    const colors = ['#1a73e8', '#4285f4', '#ea4335', '#fbbc04', '#34a853'];
    let splats = [];
    let width = 0;
    let height = 0;
    let dpr = 1;
    let animationFrame = 0;

    function resize() {
      if (animationFrame) cancelAnimationFrame(animationFrame);
      const rect = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.round(Math.max(72, width / 7));
      splats = Array.from({ length: count }, (_, i) => ({
        x: width * (.07 + Math.random() * .86),
        y: height * (.2 + Math.random() * .62),
        rx: width * (.014 + Math.random() * .032),
        ry: height * (.025 + Math.random() * .065),
        angle: Math.random() * Math.PI,
        phase: Math.random() * Math.PI * 2,
        speed: .35 + Math.random() * .65,
        color: colors[i % colors.length]
      }));
      draw(0);
    }

    function roundedFontSize() { return Math.min(height * .8, width * .235); }

    function draw(ms) {
      const t = ms / 1000;
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.globalCompositeOperation = 'source-over';

      const fontSize = roundedFontSize();
      ctx.font = `700 ${fontSize}px "Helvetica Neue", Arial, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(26,115,232,.09)';
      ctx.fillText('FlaRe', width / 2, height * .53);

      splats.forEach((s, i) => {
        const drift = reducedMotion ? 0 : Math.sin(t * s.speed + s.phase);
        const x = s.x + drift * width * .012;
        const y = s.y + Math.cos(t * s.speed * .8 + s.phase) * height * .035;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(s.angle + (reducedMotion ? 0 : t * .08 * (i % 2 ? 1 : -1)));
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, s.rx);
        gradient.addColorStop(0, `${s.color}e8`);
        gradient.addColorStop(.52, `${s.color}7a`);
        gradient.addColorStop(1, `${s.color}00`);
        ctx.scale(1, s.ry / s.rx);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, s.rx, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      ctx.globalCompositeOperation = 'destination-in';
      ctx.fillStyle = '#000';
      ctx.fillText('FlaRe', width / 2, height * .53);
      ctx.restore();

      if (!reducedMotion) animationFrame = requestAnimationFrame(draw);
    }

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
  }

  function initNavigation() {
    const nav = document.querySelector('[data-side-nav]');
    const progress = document.querySelector('[data-progress]');
    const toTop = document.querySelector('[data-to-top]');
    const topbar = document.querySelector('[data-topbar]');
    const links = [...document.querySelectorAll('[data-section]')];
    const sections = links.map(link => document.getElementById(link.dataset.section)).filter(Boolean);

    function update() {
      const y = window.scrollY;
      const viewportMiddle = y + window.innerHeight * .38;
      let active = sections[0];
      sections.forEach(section => { if (section.offsetTop <= viewportMiddle) active = section; });
      links.forEach(link => link.classList.toggle('active', link.dataset.section === active?.id));

      const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      progress.style.height = `${Math.min(100, y / max * 100)}%`;
      const visible = y > window.innerHeight * .55;
      nav.classList.toggle('visible', visible);
      toTop.classList.toggle('visible', y > window.innerHeight);
      topbar.classList.toggle('scrolled', y > 12);
    }
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    toTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' }));
    update();
  }

  function initCopy() {
    const button = document.querySelector('[data-copy-bibtex]');
    const bibtex = document.querySelector('#bibtex');
    if (!button || !bibtex) return;
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(bibtex.textContent);
        button.classList.add('copied');
        button.querySelector('span').textContent = 'Copied';
        setTimeout(() => {
          button.classList.remove('copied');
          button.querySelector('span').textContent = 'Copy BibTeX';
        }, 1800);
      } catch (_) {
        const range = document.createRange();
        range.selectNodeContents(bibtex);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
    });
  }

  initWordmark();
  initNavigation();
  initCopy();
})();

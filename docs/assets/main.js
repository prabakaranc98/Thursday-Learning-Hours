/* ── Shared helpers ─────────────────────────────────────────────────────── */

const REPO = 'https://github.com/prabakaranc98/Thursday-Learning-Hours';

/** Resolve path to sessions.json relative to current page depth */
function dataUrl() {
  // Works from both docs/ (index) and docs/ (session.html)
  const depth = location.pathname.split('/').filter(Boolean).length;
  const prefix = depth <= 1 ? '' : '../'.repeat(depth - 1);
  // When served from GitHub Pages the pathname starts with /repo-name/
  // We just need to reach the docs/data path relative to docs root
  return './data/sessions.json';
}

async function loadSessions() {
  const res = await fetch(dataUrl());
  if (!res.ok) throw new Error(`Failed to load sessions: ${res.status}`);
  return res.json();
}

/* ── SVG icons (inline, zero-dependency) ────────────────────────────────── */

const ICONS = {
  book:     `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 4h12v12H4zM4 8h12M8 4v12"/></svg>`,
  notebook: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="2" width="12" height="16" rx="1.5"/><path d="M8 2v16M4 6h4M4 10h4M4 14h4"/></svg>`,
  slides:   `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="4" width="16" height="11" rx="1.5"/><path d="M8 15l-1 3M12 15l1 3M6 18h8"/></svg>`,
  notes:    `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 3h10a1 1 0 011 1v10l-4 4H5a1 1 0 01-1-1V4a1 1 0 011-1z"/><path d="M12 14h4M7 7h6M7 10h6"/></svg>`,
  flask:    `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 3V9l-3.5 6a1 1 0 00.9 1.5h9.2a1 1 0 00.9-1.5L12 9V3"/><path d="M7 3h6"/></svg>`,
  link:     `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 5h3a3 3 0 010 6h-3M8 15H5a3 3 0 010-6h3M7 10h6"/></svg>`,
  youtube:  `<svg viewBox="0 0 20 20" fill="currentColor"><path d="M17.8 5.6a2.1 2.1 0 00-1.5-1.5C14.9 3.7 10 3.7 10 3.7s-4.9 0-6.3.4A2.1 2.1 0 002.2 5.6C1.8 7 1.8 10 1.8 10s0 3 .4 4.4a2.1 2.1 0 001.5 1.5c1.4.4 6.3.4 6.3.4s4.9 0 6.3-.4a2.1 2.1 0 001.5-1.5c.4-1.4.4-4.4.4-4.4s0-3-.4-4.4zM8.2 12.5V7.5l4.2 2.5-4.2 2.5z"/></svg>`,
  back:     `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5L7 10l5 5"/></svg>`,
  github:   `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 0C4.5 0 0 4.5 0 10a10 10 0 006.8 9.5c.5.1.7-.2.7-.5v-1.7C4.7 17.9 4.1 16 4.1 16c-.5-1.1-1.1-1.4-1.1-1.4-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.8.8.1-.6.3-1.1.6-1.3-2.2-.3-4.6-1.1-4.6-5a3.9 3.9 0 011-2.7 3.6 3.6 0 01.1-2.7s.8-.3 2.8 1a9.6 9.6 0 015 0c2-1.3 2.8-1 2.8-1 .6 1.3.2 2.3.1 2.7.6.7 1 1.6 1 2.7 0 3.9-2.4 4.7-4.6 5 .4.3.7.9.7 1.8v2.7c0 .3.2.6.7.5A10 10 0 0020 10C20 4.5 15.5 0 10 0z" clip-rule="evenodd"/></svg>`,
  external: `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M11 3h6v6M17 3l-8 8M9 5H4a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1v-5"/></svg>`,
};

/* ── Status helpers ─────────────────────────────────────────────────────── */

const STATUS_LABEL = { upcoming: 'Upcoming', recorded: 'Recorded', live: 'Live Now' };

function statusBadge(status) {
  return `<span class="status-badge status-${status}">${STATUS_LABEL[status] || status}</span>`;
}

/* ── Pill link ──────────────────────────────────────────────────────────── */

function resPill(icon, label, url) {
  if (!url) return '';
  return `<a class="res-pill" href="${url}" target="_blank" rel="noopener">
    ${ICONS[icon] || ''}${label}
  </a>`;
}

/* ── Card resource row ──────────────────────────────────────────────────── */

function cardResources(s) {
  const r = s.resources || {};
  const pills = [
    r.notes         && resPill('notes',   'Notes',       r.notes.url),
    r.reading       && resPill('book',    'Reading',     r.reading.url),
    r.slides        && resPill('slides',  'Slides',      r.slides.url),
    r.resources_link && resPill('link',   'Resources',   r.resources_link.url),
    s.youtube_id    && resPill('youtube', 'Watch',
                        `https://www.youtube.com/watch?v=${s.youtube_id}`),
    (r.experiments || []).length &&
      resPill('flask', `${r.experiments.length} Demo${r.experiments.length > 1 ? 's' : ''}`,
              `session.html?id=${s.id}`),
  ].filter(Boolean);

  if (!pills.length) return '';
  return `<div class="card-resources">${pills.join('')}</div>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   INDEX PAGE — session card grid
══════════════════════════════════════════════════════════════════════════ */

function renderIndex(sessions) {
  const grid = document.getElementById('sessions-grid');
  const count = document.getElementById('sessions-count');
  if (!grid) return;

  if (count) count.textContent = `${sessions.length} session${sessions.length !== 1 ? 's' : ''}`;

  // Stats
  const recorded = sessions.filter(s => s.status === 'recorded').length;
  const upcoming = sessions.filter(s => s.status === 'upcoming').length;
  const el = id => document.getElementById(id);
  if (el('stat-total'))    el('stat-total').textContent    = sessions.length;
  if (el('stat-recorded')) el('stat-recorded').textContent = recorded;
  if (el('stat-upcoming')) el('stat-upcoming').textContent = upcoming;

  grid.innerHTML = sessions.map(s => `
    <a class="session-card" href="session.html?id=${s.id}">
      <div class="card-header">
        <div class="session-number">S${s.number}</div>
        <div class="card-meta">
          <div class="card-title">${s.title}</div>
          <div class="card-subtitle">${s.subtitle || ''}</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;gap:.5rem;">
        ${statusBadge(s.status)}
        <span style="font-size:.8rem;color:var(--text-muted)">${s.date || ''}</span>
      </div>
      <p class="card-desc">${s.description}</p>
      <div class="tag-list">${(s.tags || []).map(t => `<span class="tag">${t}</span>`).join('')}</div>
      ${cardResources(s)}
    </a>
  `).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   SESSION DETAIL PAGE
══════════════════════════════════════════════════════════════════════════ */

function btnResource(label, url, icon) {
  if (!url) return '';
  return `<a class="btn-resource" href="${url}" target="_blank" rel="noopener">
    ${ICONS[icon] || ICONS.external} ${label}
  </a>`;
}

function renderDetailRow(name, desc, type, url) {
  return `
    <div class="res-link-row">
      <div>
        <div class="res-link-name">${name}</div>
        ${desc ? `<div class="res-link-desc">${desc}</div>` : ''}
      </div>
      <div style="display:flex;align-items:center;gap:.5rem;flex-shrink:0">
        ${type ? `<span class="res-link-type">${type}</span>` : ''}
        ${btnResource('Open', url, 'external')}
      </div>
    </div>`;
}

function resourceBlock(icon, title, bodyHTML) {
  return `
    <div class="resource-block">
      <div class="resource-block-header">
        <span class="resource-block-icon">${icon}</span>
        <span class="resource-block-title">${title}</span>
      </div>
      <div class="resource-block-body">${bodyHTML}</div>
    </div>`;
}

function renderSession(session) {
  const wrap = document.getElementById('detail-wrap');
  if (!wrap) return;

  const r = session.resources || {};

  // YouTube
  let ytHTML;
  if (session.youtube_id) {
    ytHTML = `<div class="yt-embed-wrap">
      <iframe src="https://www.youtube-nocookie.com/embed/${session.youtube_id}"
        title="YouTube video player" allow="accelerometer; autoplay; clipboard-write;
        encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>`;
  } else {
    ytHTML = `<div class="yt-placeholder">
      ${ICONS.youtube}
      <p>Recording not yet available — check back after the session!</p>
    </div>`;
  }

  // Quick-access pills
  const quickLinks = [
    r.notes          && resPill('notes',   'Notes',      r.notes.url),
    r.reading        && resPill('book',    'Reading',    r.reading.url),
    r.slides         && resPill('slides',  'Slides',     r.slides.url),
    r.resources_link && resPill('link',    'Resources',  r.resources_link.url),
    r.report         && resPill('notebook','Report',     r.report.url),
    session.youtube_id && resPill('youtube', 'Watch',
      `https://www.youtube.com/watch?v=${session.youtube_id}`),
  ].filter(Boolean).join('');

  // Experiments block
  const experiments = r.experiments || [];
  const expBody = experiments.length
    ? experiments.map(e => renderDetailRow(e.name, e.description, e.type, e.url)).join('')
    : '<p style="color:var(--text-muted);font-size:.875rem">No experiments yet.</p>';

  // Reading block
  const readingBody = r.reading
    ? renderDetailRow(r.reading.label, 'Annotated reading list with 5 themed sections and suggested order.', 'markdown', r.reading.url)
    : '';

  // Resources block
  const resourcesBody = r.resources_link
    ? renderDetailRow(r.resources_link.label, 'Frameworks, datasets, libraries, and courses.', 'markdown', r.resources_link.url)
    : '';

  // Slides block
  const slidesBody = r.slides
    ? renderDetailRow(r.slides.label, 'Full session deck outline for the 60-min session.', 'markdown', r.slides.url)
    : '';

  // Notes block
  const notesBody = r.notes
    ? renderDetailRow(r.notes.label, 'Session notes, diagrams, key takeaways, and discussion questions.', 'markdown', r.notes.url)
    : '';

  // Report block
  const reportBody = r.report
    ? renderDetailRow(r.report.label, 'Formal LaTeX technical report with equations and bibliography.', 'latex', r.report.url)
    : '';

  // GitHub link
  const ghFolderUrl = session.github_folder
    ? `${REPO}/tree/main/${session.github_folder}` : REPO;

  wrap.innerHTML = `
    <a class="back-link" href="index.html">${ICONS.back} All Sessions</a>

    <div class="detail-header">
      <div class="detail-number-row">
        <span class="detail-number-badge">Session ${session.number}</span>
        ${statusBadge(session.status)}
        <span class="detail-date">${session.date || 'Date TBD'}</span>
      </div>
      <h1 class="detail-title">${session.title}</h1>
      <p class="detail-subtitle">${session.subtitle || ''}</p>
      <p class="detail-desc">${session.description}</p>
      <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-bottom:.75rem">
        <div class="tag-list">${(session.tags || []).map(t => `<span class="tag">${t}</span>`).join('')}</div>
      </div>
      <div class="tag-list">${quickLinks}</div>
    </div>

    <div class="yt-section">${ytHTML}</div>

    <div class="resource-sections">
      ${slidesBody   ? resourceBlock('📊', 'Slides',        slidesBody)   : ''}
      ${notesBody    ? resourceBlock('📝', 'Session Notes', notesBody)    : ''}
      ${readingBody  ? resourceBlock('📚', 'Reading List',  readingBody)  : ''}
      ${expBody      ? resourceBlock('🧪', 'Experiments & Demos', expBody) : ''}
      ${resourcesBody? resourceBlock('🔗', 'Resources & Tools', resourcesBody) : ''}
      ${reportBody   ? resourceBlock('📄', 'Technical Report', reportBody) : ''}
      ${resourceBlock('💾', 'Source Files', `
        <div class="res-link-row">
          <div>
            <div class="res-link-name">GitHub Folder</div>
            <div class="res-link-desc">All session files: code, tex, slides, notes.</div>
          </div>
          ${btnResource('View on GitHub', ghFolderUrl, 'github')}
        </div>`)}
    </div>`;

  document.title = `TLH-${session.number} · ${session.title}`;
}

/* ══════════════════════════════════════════════════════════════════════════
   ROUTER — detect which page we're on and initialise
══════════════════════════════════════════════════════════════════════════ */

async function init() {
  const isSession = document.getElementById('detail-wrap') !== null;

  try {
    const sessions = await loadSessions();

    if (isSession) {
      // session.html?id=tlh-1
      const id = new URLSearchParams(location.search).get('id');
      const session = sessions.find(s => s.id === id);
      if (session) {
        renderSession(session);
      } else {
        const wrap = document.getElementById('detail-wrap');
        if (wrap) wrap.innerHTML = `<div class="state-msg">Session not found. <a href="index.html">Go back</a></div>`;
      }
    } else {
      renderIndex(sessions);
    }
  } catch (err) {
    console.error(err);
    const target = document.getElementById('sessions-grid') || document.getElementById('detail-wrap');
    if (target) target.innerHTML = `<div class="state-msg">Could not load sessions data.<br><small>${err.message}</small></div>`;
  }
}

document.addEventListener('DOMContentLoaded', init);

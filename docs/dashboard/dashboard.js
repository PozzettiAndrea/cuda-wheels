// --- Wheel list modal ---

function formatBytes(bytes) {
  if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + ' GB';
  if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + ' MB';
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return bytes + ' B';
}

function escapeHtml(str) {
  var div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function parseMetadata(raw) {
  var fields = {};
  var lines = raw.split('\n');
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    // Stop at the body separator (blank line after headers)
    if (line.trim() === '') break;
    var colon = line.indexOf(': ');
    if (colon === -1) continue;
    var key = line.substring(0, colon);
    var val = line.substring(colon + 2);
    // Requires-Dist can appear multiple times
    if (key === 'Requires-Dist') {
      if (!fields[key]) fields[key] = [];
      fields[key].push(val);
    } else if (!fields[key]) {
      fields[key] = val;
    }
  }
  return fields;
}

function openModal(pkg) {
  var wheels = window.__WHEEL_DATA__[pkg];
  if (!wheels) return;
  document.getElementById('modal-title').textContent = pkg + ' (' + wheels.length + ' wheels)';
  var tbody = document.getElementById('modal-wheels');
  tbody.innerHTML = wheels.map(function(w, idx) {
    var nameCell = w.url ? '<a href="' + w.url + '">' + w.name + '</a>' : w.name;
    var contentsBtn = w.contents
      ? '<button class="contents-btn" onclick="inspectWheel(\'' + pkg + '\',' + idx + ')">Inspect</button>'
      : '-';
    var buildTime = w.build_time || '-';
    return '<tr><td>' + nameCell + '</td><td>' + w.size + '</td><td>' + buildTime + '</td><td>' + contentsBtn + '</td></tr>';
  }).join('');
  document.getElementById('modal').classList.add('active');
}

function closeModal() {
  document.getElementById('modal').classList.remove('active');
  if (window.location.hash.indexOf('pkg=') !== -1) {
    history.pushState(null, '', window.location.pathname);
  }
}

// --- Wheel contents modal ---

function inspectWheel(pkg, idx) {
  var wheel = window.__WHEEL_DATA__[pkg][idx];
  if (!wheel || !wheel.contents) return;

  var modal = document.getElementById('contents-modal');
  var tbody = document.getElementById('contents-files');
  document.getElementById('contents-title').textContent = wheel.name;
  document.getElementById('contents-loading').style.display = 'none';

  // Render metadata section
  var metaDiv = document.getElementById('contents-metadata');
  if (wheel.metadata) {
    var fields = parseMetadata(wheel.metadata);
    var html = '<table class="metadata-table">';
    ['Name', 'Version', 'Summary', 'Requires-Python'].forEach(function(key) {
      if (fields[key]) {
        html += '<tr><td class="meta-key">' + escapeHtml(key) + '</td><td>' + escapeHtml(fields[key]) + '</td></tr>';
      }
    });
    if (fields['Requires-Dist'] && fields['Requires-Dist'].length) {
      html += '<tr><td class="meta-key">Requires-Dist</td><td>';
      fields['Requires-Dist'].forEach(function(dep) {
        var cls = dep.indexOf('git+') !== -1 ? ' class="meta-warn"' : '';
        html += '<div' + cls + '>' + escapeHtml(dep) + '</div>';
      });
      html += '</td></tr>';
    }
    html += '</table>';
    metaDiv.innerHTML = html;
    metaDiv.style.display = 'block';
  } else {
    metaDiv.innerHTML = '';
    metaDiv.style.display = 'none';
  }

  var files = wheel.contents;
  tbody.innerHTML = files.map(function(f) {
    var icon = f.dir ? '&#x1F4C1;' : '';
    var sizeStr = f.dir ? '-' : formatBytes(f.size);
    return '<tr><td>' + icon + ' ' + f.path + '</td><td>' + sizeStr + '</td></tr>';
  }).join('');

  modal.classList.add('active');
  history.pushState(null, '', '#wheel=' + encodeURIComponent(wheel.name));
}

function closeContentsModal() {
  document.getElementById('contents-modal').classList.remove('active');
  if (window.location.hash.indexOf('wheel=') !== -1) {
    history.back();
  }
}

// --- Hash routing ---

function checkHash() {
  var hash = window.location.hash;
  var pkgMatch = hash.match(/^#pkg=(.+)$/);
  if (pkgMatch) {
    openModal(decodeURIComponent(pkgMatch[1]));
    return;
  }
  document.getElementById('modal').classList.remove('active');
}

// Close modals on overlay click
document.getElementById('modal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});
document.getElementById('contents-modal').addEventListener('click', function(e) {
  if (e.target === this) closeContentsModal();
});

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    if (document.getElementById('contents-modal').classList.contains('active')) {
      closeContentsModal();
    } else {
      closeModal();
    }
  }
});

window.addEventListener('hashchange', checkHash);
checkHash();

// --- Tab switching ---

document.querySelectorAll('.tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.tab-content').forEach(function(c) { c.style.display = 'none'; });
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).style.display = 'block';
    if (tab.dataset.tab === 'missing' && !window._missingRendered) {
      renderMissingTables();
    }
  });
});

// --- Missing tab ---

function renderMissingTables(filterPkg) {
  var data = window.__MISSING_DATA__;
  if (!data) return;
  window._missingRendered = true;

  // Populate filter dropdown
  var select = document.getElementById('missing-pkg-filter');
  if (select.options.length <= 1) {
    Object.keys(data).sort().forEach(function(pkg) {
      var opt = document.createElement('option');
      opt.value = pkg;
      opt.textContent = pkg + ' (' + data[pkg].missing.length + ' missing)';
      select.appendChild(opt);
    });
    select.addEventListener('change', function() {
      renderMissingTables(this.value);
    });
  }

  var container = document.getElementById('missing-tables');
  var html = '';

  var pkgs = filterPkg ? [filterPkg] : Object.keys(data).sort();
  pkgs.forEach(function(pkg) {
    var info = data[pkg];
    if (!info) return;
    var pct = info.expected > 0 ? Math.round(100 * info.built / info.expected) : 100;
    var isComplete = info.missing.length === 0;

    html += '<div class="missing-pkg">';
    html += '<h3>' + escapeHtml(pkg) + '</h3>';
    html += '<div class="missing-summary">';
    if (isComplete) {
      html += '<span class="complete">' + info.built + ' / ' + info.expected + ' built (100%)</span>';
    } else {
      html += info.built + ' / ' + info.expected + ' built (' + pct + '%) &mdash; <strong>' + info.missing.length + ' missing</strong>';
    }
    html += '</div>';

    if (!isComplete) {
      html += buildMatrix(info);
    }
    html += '</div>';
  });

  container.innerHTML = html;
}

function buildMatrix(info) {
  // Build sets of missing combos for fast lookup
  var missingSet = {};
  info.missing.forEach(function(m) {
    missingSet[m.cuda + '|' + m.torch + '|' + m.python + '|' + m.platform] = true;
  });

  // Collect unique row keys (cuda+torch) and column keys (python+platform)
  var rowKeys = {};
  var colKeys = {};

  // We need the full expected matrix, not just missing entries.
  // Reconstruct from: all combos = built (expected - missing) + missing
  // Since we only have missing entries, we need all combos.
  // The missing data includes expected count but not all combos explicitly.
  // We'll show only missing entries in a flat table instead.

  // Actually let's build a proper matrix. We need all (cuda,torch) × (python,platform) combos.
  // From missing entries we can infer all columns and rows that have at least one missing.
  // But to show the full picture we need all combos. Let's just list missing as a table.

  // Flat table approach: group by cuda+torch, show missing python+platform
  var groups = {};
  info.missing.forEach(function(m) {
    var key = 'cu' + m.cuda.replace('.','') + ' torch ' + m.torch;
    if (!groups[key]) groups[key] = [];
    groups[key].push('py' + m.python + ' ' + m.platform);
  });

  var html = '<table class="matrix-table"><thead><tr><th>CUDA / PyTorch</th><th>Missing combos</th></tr></thead><tbody>';
  Object.keys(groups).sort().forEach(function(key) {
    html += '<tr><td class="row-header">' + escapeHtml(key) + '</td>';
    html += '<td>' + groups[key].sort().map(function(c) {
      return '<span class="cell-miss">' + escapeHtml(c) + '</span>';
    }).join(', ') + '</td></tr>';
  });
  html += '</tbody></table>';
  return '<div class="matrix-wrap">' + html + '</div>';
}

// Handle #tab=missing in URL
(function() {
  if (window.location.hash === '#tab=missing') {
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.tab-content').forEach(function(c) { c.style.display = 'none'; });
    var missingTab = document.querySelector('[data-tab="missing"]');
    if (missingTab) {
      missingTab.classList.add('active');
      document.getElementById('tab-missing').style.display = 'block';
      renderMissingTables();
    }
  }
})();

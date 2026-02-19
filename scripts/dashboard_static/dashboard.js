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

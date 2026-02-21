// --- Wheel filename parsing ---

function parseWheelName(name) {
  // e.g. cc_torch-0.2+cu124torch25-cp310-cp310-manylinux_2_35_x86_64.whl
  var m = name.match(/^([^-]+)-([^-]+)\+cu(\d+)torch(\d+)-cp(\d+)-[^-]+-(.+)\.whl$/);
  if (!m) return null;
  var cu = m[3], torch = m[4], py = m[5], plat = m[6];
  // CUDA: "124" → "12.4", "130" → "13.0" (always last digit is minor)
  // Torch: "25" → "2.5", "210" → "2.10" (first digit is major, rest is minor)
  // Python: "310" → "3.10", "313" → "3.13" (first digit is major, rest is minor)
  return {
    os: (plat.indexOf('linux') !== -1 || plat.indexOf('manylinux') !== -1) ? 'Linux' : 'Windows',
    cuda: cu.slice(0, -1) + '.' + cu.slice(-1),
    torch: torch[0] + '.' + torch.slice(1),
    python: py[0] + '.' + py.slice(1)
  };
}

// --- State ---

var wheels = window.__INSTALL_DATA__ || {};
var allPkgs = Object.keys(wheels).sort();
var selected = { os: '', cuda: '', torch: '', python: '' };
var selectedPkgs = {};

// Pre-parse all wheel metadata
var parsed = {};
allPkgs.forEach(function(pkg) {
  parsed[pkg] = wheels[pkg].map(function(w) {
    var info = parseWheelName(w.n);
    return { name: w.n, url: w.u, info: info };
  }).filter(function(w) { return w.info !== null; });
});

// --- Helpers ---

function unique(arr) {
  var seen = {};
  return arr.filter(function(v) { return seen[v] ? false : (seen[v] = true); });
}

function versionSort(a, b) {
  var pa = a.split('.').map(Number), pb = b.split('.').map(Number);
  for (var i = 0; i < Math.max(pa.length, pb.length); i++) {
    var va = pa[i] || 0, vb = pb[i] || 0;
    if (va !== vb) return va - vb;
  }
  return 0;
}

function getFilteredWheels(pkg, criteria) {
  return parsed[pkg].filter(function(w) {
    if (criteria.os && w.info.os !== criteria.os) return false;
    if (criteria.cuda && w.info.cuda !== criteria.cuda) return false;
    if (criteria.torch && w.info.torch !== criteria.torch) return false;
    if (criteria.python && w.info.python !== criteria.python) return false;
    return true;
  });
}

function getAllValues(field, criteria) {
  var vals = [];
  allPkgs.forEach(function(pkg) {
    getFilteredWheels(pkg, criteria).forEach(function(w) {
      vals.push(w.info[field]);
    });
  });
  return unique(vals).sort(versionSort);
}

// --- Dropdown population ---

function populateSelect(id, values, placeholder) {
  var sel = document.getElementById(id);
  var current = sel.value;
  sel.innerHTML = '<option value="">' + placeholder + '</option>';
  values.forEach(function(v) {
    sel.innerHTML += '<option value="' + v + '"' + (v === current ? ' selected' : '') + '>' + v + '</option>';
  });
  sel.disabled = values.length === 0;
  // Keep current value if still valid
  if (current && values.indexOf(current) !== -1) {
    sel.value = current;
  } else {
    sel.value = '';
  }
  return sel.value;
}

function updateDropdowns() {
  // OS - always available
  var osValues = getAllValues('os', {});
  selected.os = populateSelect('sel-os', osValues, 'Select...');

  // CUDA - filtered by OS
  var cudaFilter = { os: selected.os || undefined };
  var cudaValues = getAllValues('cuda', cudaFilter);
  selected.cuda = populateSelect('sel-cuda', cudaValues, selected.os ? 'Select...' : 'Select OS first');

  // Torch - filtered by OS + CUDA
  var torchFilter = { os: selected.os || undefined, cuda: selected.cuda || undefined };
  var torchValues = getAllValues('torch', torchFilter);
  selected.torch = populateSelect('sel-torch', torchValues, selected.cuda ? 'Select...' : 'Select CUDA first');

  // Python - filtered by OS + CUDA + Torch
  var pyFilter = { os: selected.os || undefined, cuda: selected.cuda || undefined, torch: selected.torch || undefined };
  var pyValues = getAllValues('python', pyFilter);
  selected.python = populateSelect('sel-python', pyValues, selected.torch ? 'Select...' : 'Select PyTorch first');

  updatePackageChips();
  updateCommand();
}

// --- Package chips ---

function updatePackageChips() {
  var container = document.getElementById('pkg-chips');
  var html = '';
  var envSelected = selected.os && selected.cuda && selected.torch && selected.python;

  allPkgs.forEach(function(pkg) {
    var display = pkg;
    if (envSelected) {
      var matches = getFilteredWheels(pkg, selected);
      if (matches.length === 0) {
        html += '<span class="pkg-chip unavailable">' + display + '</span>';
        // Deselect if no longer available
        delete selectedPkgs[pkg];
        return;
      }
    }
    var isSelected = selectedPkgs[pkg];
    var cls = 'pkg-chip' + (isSelected ? ' selected' : '') + (!envSelected ? ' unavailable' : '');
    html += '<span class="' + cls + '" data-pkg="' + pkg + '" onclick="togglePkg(\'' + pkg + '\')">' + display + '</span>';
  });
  container.innerHTML = html;
}

function togglePkg(pkg) {
  if (!(selected.os && selected.cuda && selected.torch && selected.python)) return;
  var matches = getFilteredWheels(pkg, selected);
  if (matches.length === 0) return;

  if (selectedPkgs[pkg]) {
    delete selectedPkgs[pkg];
  } else {
    selectedPkgs[pkg] = true;
  }
  updatePackageChips();
  updateCommand();
}

// --- Command generation ---

function updateCommand() {
  var box = document.getElementById('command-box');
  var copyBtn = document.getElementById('copy-btn');
  var warn = document.getElementById('warn');
  warn.style.display = 'none';

  var pkgs = Object.keys(selectedPkgs).sort();
  if (pkgs.length === 0 || !selected.os || !selected.cuda || !selected.torch || !selected.python) {
    box.innerHTML = '<span class="placeholder">Select your environment above, then click packages to install.</span>';
    copyBtn.style.display = 'none';
    return;
  }

  var urls = [];
  var missing = [];
  pkgs.forEach(function(pkg) {
    var matches = getFilteredWheels(pkg, selected);
    if (matches.length > 0) {
      urls.push(matches[0].url);
    } else {
      missing.push(pkg);
    }
  });

  if (urls.length === 0) {
    box.innerHTML = '<span class="placeholder">No wheels found for the selected combination.</span>';
    copyBtn.style.display = 'none';
    return;
  }

  var cmd = 'pip install \\\n' + urls.map(function(u) {
    return '  "' + u + '"';
  }).join(' \\\n');

  box.textContent = cmd;
  box.appendChild(copyBtn);
  copyBtn.style.display = 'block';

  if (missing.length > 0) {
    warn.textContent = 'No wheel found for: ' + missing.join(', ');
    warn.style.display = 'block';
  }
}

function copyCommand() {
  var box = document.getElementById('command-box');
  var btn = document.getElementById('copy-btn');
  // Get text without the button text
  var text = box.textContent.replace('Copy', '').replace('Copied!', '').trim();
  navigator.clipboard.writeText(text).then(function() {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(function() {
      btn.textContent = 'Copy';
      btn.classList.remove('copied');
    }, 2000);
  });
}

// --- Event listeners ---

document.getElementById('sel-os').addEventListener('change', function() {
  selected.os = this.value;
  selected.cuda = '';
  selected.torch = '';
  selected.python = '';
  selectedPkgs = {};
  updateDropdowns();
});

document.getElementById('sel-cuda').addEventListener('change', function() {
  selected.cuda = this.value;
  selected.torch = '';
  selected.python = '';
  selectedPkgs = {};
  updateDropdowns();
});

document.getElementById('sel-torch').addEventListener('change', function() {
  selected.torch = this.value;
  selected.python = '';
  selectedPkgs = {};
  updateDropdowns();
});

document.getElementById('sel-python').addEventListener('change', function() {
  selected.python = this.value;
  updatePackageChips();
  updateCommand();
});

// --- Init ---
updateDropdowns();

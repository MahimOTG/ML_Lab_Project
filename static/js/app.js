document.addEventListener('DOMContentLoaded', function () {
  var tabButtons = document.querySelectorAll('.tab-button');
  var panels = {
    manual: document.getElementById('manual-panel'),
    upload: document.getElementById('upload-panel')
  };

  tabButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      var tab = button.dataset.tab;
      tabButtons.forEach(function (item) {
        item.classList.toggle('active', item === button);
      });
      Object.keys(panels).forEach(function (name) {
        if (panels[name]) {
          panels[name].classList.toggle('hidden', name !== tab);
        }
      });
    });
  });

  var datasetSelect = document.getElementById('dataset');
  if (datasetSelect) {
    datasetSelect.addEventListener('change', function () {
      var active = document.querySelector('.tab-button.active');
      var tab = active ? active.dataset.tab : 'manual';
      window.location.href = '/?dataset=' + encodeURIComponent(datasetSelect.value) +
        '&tab=' + tab + '#input-engine';
    });
  }

  var toggle = document.getElementById('sensor-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var expanded = toggle.dataset.expanded === 'true';
      var next = !expanded;
      document.querySelectorAll('.sensor-extra').forEach(function (field) {
        field.classList.toggle('hidden', !next);
      });
      toggle.dataset.expanded = String(next);
      toggle.classList.toggle('expanded', next);
      var label = toggle.querySelector('span');
      if (label) {
        label.textContent = next ? 'Hide Extra Sensors' : 'Show All ' + (toggle.dataset.total || '21') + ' Sensors';
      }
    });
  }
});

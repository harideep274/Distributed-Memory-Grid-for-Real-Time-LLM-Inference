const runButton = document.querySelector('#run-demo');
const log = document.querySelector('#demo-log');
const state = document.querySelector('#demo-state');

const steps = [
  ['> request accepted · prompt queued at coordinator', 'log-ok'],
  ['> scheduler admits request into active decode set', ''],
  ['> worker-01 · local weight shard ready', ''],
  ['> worker-02 · local weight shard ready', ''],
  ['> partial outputs gathered · reference path only', 'log-warn'],
  ['> token event streamed · demo complete', 'log-ok'],
];

runButton?.addEventListener('click', async () => {
  if (runButton.disabled) return;
  runButton.disabled = true;
  runButton.innerHTML = 'Running visual demo <span>⟳</span>';
  state.textContent = 'RUNNING';
  log.replaceChildren();
  for (const [message, className] of steps) {
    await new Promise((resolve) => window.setTimeout(resolve, 480));
    const line = document.createElement('p');
    line.textContent = message;
    if (className) line.className = className;
    log.append(line);
  }
  state.textContent = 'SIMULATED';
  runButton.disabled = false;
  runButton.innerHTML = 'Run again <span>↻</span>';
});

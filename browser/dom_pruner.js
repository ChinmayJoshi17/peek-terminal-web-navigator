// Browser DOM Pruner - Interactive Element Highlighting Utility

window.getInteractiveElements = function() {
  const interactiveSelectors = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[role="button"][onclick]',
    '[role="button"]:not([aria-disabled="true"])',
    '[role="searchbox"]',
    '[role="combobox"]',
    '[role="link"]',
    '[role="menuitem"]',
    '[role="tab"]'
  ];

  const allElements = Array.from(document.querySelectorAll(interactiveSelectors.join(',')));
  const visibleInteractiveElements = [];
  let idCounter = 1;

  allElements.forEach(element => {
    if (element.offsetWidth > 0 && element.offsetHeight > 0) {
      const computedStyle = window.getComputedStyle(element);
      if (
        computedStyle.display !== 'none' &&
        computedStyle.visibility !== 'hidden' &&
        computedStyle.opacity !== '0'
      ) {
        const agentId = `i${idCounter++}`;
        element.setAttribute('data-agent-id', agentId);

        // Create floating badge (hidden visually — display:none keeps DOM intact for debugging)
        const badge = document.createElement('div');
        badge.className = '__agent_badge__';
        badge.textContent = agentId;
        badge.style.cssText = `
          display: none;
        `;

        document.body.appendChild(badge);

        const textContent = (element.innerText || element.value || '').trim();
        const placeholder = element.placeholder || element.getAttribute('placeholder') || '';
        const role = element.getAttribute('role') || '';
        const ariaLabel = element.getAttribute('aria-label') || element.getAttribute('aria-labelledby') || '';
        const href = element.getAttribute('href') || '';
        const name = element.getAttribute('name') || '';
        const cursor = computedStyle.cursor || '';

        visibleInteractiveElements.push({
          id: agentId,
          tag: element.tagName.toLowerCase(),
          text: textContent.slice(0, 80),
          type: element.type || '',
          placeholder: placeholder.slice(0, 60),
          role: role,
          aria_label: ariaLabel.slice(0, 60),
          href: href.slice(0, 100),
          name: name.slice(0, 40),
          cursor: cursor
        });
      }
    }
  });

  return visibleInteractiveElements;
};

window.cleanBadges = function() {
  // 1. Remove ONLY the injected badge divs
  document.querySelectorAll('.__agent_badge__').forEach(badge => badge.remove());

  // 2. Clear the attribute from the actual page elements (without deleting them)
  document.querySelectorAll('[data-agent-id]').forEach(el => el.removeAttribute('data-agent-id'));
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    getInteractiveElements: window.getInteractiveElements,
    cleanBadges: window.cleanBadges
  };
}

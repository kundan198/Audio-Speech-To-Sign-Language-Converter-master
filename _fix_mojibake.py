from pathlib import Path
import re
files = [
    Path('templates/animation.html'),
    Path('templates/conversation.html'),
    Path('templates/sign_to_text.html'),
]
replacements = {
    'Ã‚Â·': ' - ',
    'Â·': ' - ',
    'Ã¢â‚¬Â¦': '...',
    'â€¦': '...',
    'Ã¢â‚¬â€': '-',
    'â€”': '-',
    'â€“': '-',
    'Ã¢Å¡Â ': '!',
    'âš ': '!',
    'Ã¢Å“â€”': 'x',
    'âœ—': 'x',
    'âœ“': 'ok',
    'âŒ': 'x',
    'â†’': '->',
    'â†': '<-',
    'Ã°Å¸Å½â„¢ ': '',
    'ðŸ”': 'Scanning',
    'ðŸ“¸': 'Snap',
    'â³': 'Hold',
    'âœ‹': 'Hand',
    'Ã¢â€ Â³': '->',
}
for p in files:
    s = p.read_text(encoding='utf-8', errors='replace')
    for a,b in replacements.items():
        s = s.replace(a,b)
    s = re.sub(r'â”€{4,}', '----------------', s)
    p.write_text(s, encoding='utf-8')
print('cleaned', len(files), 'files')

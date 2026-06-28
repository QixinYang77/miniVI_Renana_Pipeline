// Intentionally empty.
//
// The Dash callback rebuilds threshold lines with fixed x0/x1 after every
// threshold edit. Earlier versions corrected horizontal shape movement with a
// front-end relayout call, but that could race Plotly's own shape-drag event
// and make the red line snap back vertically. Keeping this asset as a no-op
// avoids stale browser caches for the old file name while removing that race.

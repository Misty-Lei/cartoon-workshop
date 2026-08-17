import templateCatalog from './template-catalog.json'

export const templates = templateCatalog.map(template => ({
  ...template,
  imageUrl: `${import.meta.env.BASE_URL}${template.imageUrl.replace(/^\//, '')}`,
}))

export function svgUrl(svg) {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

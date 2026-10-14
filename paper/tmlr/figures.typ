// Reuse canonical vector drawings; captions are converted separately in LaTeX.
#let doc = include "../main.typ"
#let find(it) = {
  if it.func() == figure { (it,) }
  else {
    let fields = it.fields()
    if "children" in fields { fields.children.map(find).flatten() }
    else if "child" in fields { find(fields.child) }
    else if "body" in fields and type(fields.body) == content { find(fields.body) }
    else { () }
  }
}
#set page(width: 470pt, height: auto, margin: 1pt)
#set text(size: 10pt)
#for it in find(doc).filter(it => it.has("kind") and it.kind == image) {
  it.body
  pagebreak()
}

// Native JSON keeps evaluated math and dynamic tables; disambiguate list kinds.
#let doc = include "../main.typ"
#let encode(it) = {
  let fields = it.fields()
  let name = if it.func() == enum.item { "enum-item" } else if it.func() == list.item { "list-item" } else { repr(it.func()) }
  let result = (func: name)
  for (k, v) in fields {
    if k in ("body", "child") and type(v) == content { result.insert(k, encode(v)) }
    else if k == "children" { result.insert(k, v.map(encode)) }
    else { result.insert(k, v) }
  }
  result
}
#metadata(encode(doc)) <manuscript>

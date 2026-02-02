import { c as comment, a as append } from "../chunks/BpcVB9B6.js";
import "../chunks/Dl1I_g4l.js";
import { o as hydrating, q as hydrate_next, j as first_child } from "../chunks/NVQkUsao.js";
function slot(anchor, $$props, name, slot_props, fallback_fn) {
  if (hydrating) {
    hydrate_next();
  }
  var slot_fn = $$props.$$slots?.[name];
  var is_interop = false;
  if (slot_fn === true) {
    slot_fn = $$props["children"];
    is_interop = true;
  }
  if (slot_fn === void 0) ;
  else {
    slot_fn(anchor, is_interop ? () => slot_props : slot_props);
  }
}
const ssr = true;
const prerender = true;
const _layout$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  prerender,
  ssr
}, Symbol.toStringTag, { value: "Module" }));
function _layout($$anchor, $$props) {
  var fragment = comment();
  var node = first_child(fragment);
  slot(node, $$props, "default", {});
  append($$anchor, fragment);
}
export {
  _layout as component,
  _layout$1 as universal
};
//# sourceMappingURL=0.4eZn7VH2.js.map

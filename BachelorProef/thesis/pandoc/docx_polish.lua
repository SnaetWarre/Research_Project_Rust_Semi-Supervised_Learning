--- Pandoc Lua filter: nicer .docx for thesis builds
--- - Page break before every level-1 heading except the first (title page block stays together).
--- - Default figure width so SVGs/charts do not spill past margins (override per image with {width=...}).

local PAGE_BREAK = pandoc.RawBlock("openxml", "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

local h1_count = 0

function Header(el)
  if el.level == 1 then
    h1_count = h1_count + 1
    if h1_count > 1 then
      return { PAGE_BREAK, el }
    end
  end
  return el
end

function Image(img)
  local attrs = img.attr.attributes
  local w = attrs.width
  if w == nil or w == "" then
    attrs.width = "5.9in"
  end
  return img
end

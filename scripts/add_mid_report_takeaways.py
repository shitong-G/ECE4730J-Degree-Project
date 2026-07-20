from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


PPTX_IN = Path("mid_report1.pptx")
PPTX_OUT = Path("mid_report1_with_takeaways.pptx")

EMU = 914400
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


TAKEAWAYS = {
    1: "Key message: 通过 scene reuse + thermal control 提升 Pi 上 RT-DETR 的持续检测能力。",
    2: "结论：baseline 长时运行升至约 85C，FPS 约从 0.32 降到 0.26（-19%）。",
    3: "结论：640/480/320 三档配置提供 quality-speed-thermal 的动态调节空间。",
    4: "结论：优先提高 inference interval，真正 throttling 时再降到 480p，尽量保护检测质量。",
    5: "结论：dashboard 可远程观察温度、FPS、latency、tracking 和 controller decision。",
    6: "结论：ablation 将 scene reuse、thermal control 和 co-adaptation 的贡献分开验证。",
    7: "结论：LK 将 output FPS 从 0.353 提升到 1.485（+321%），约 98% 帧由 tracking 输出。",
    8: "结论：thermal control 将平均温度从 82.7C 降到 74.6C（-9.8%），但 FPS 基本不提升。",
    9: "结论：co-adaptive 保持 output FPS +315% vs baseline，同时平均温度降低约 11%。",
    10: "结论：综合最优约为 1.46 output FPS、73.5C mean temp、0.23% throttled。",
    11: "结论：CSI camera + PWM fan 让系统从离线实验走向真实硬件闭环。",
    12: "结论：live demo 验证 camera -> inference -> tracking -> dashboard 在线链路已打通。",
    13: "Next: 完善 workload classification，并在更多真实场景中验证稳定性。",
}

APPEND_SLIDES = set(range(1, 12))
TEXTBOX_SLIDES = {12, 13}


def qn(tag: str) -> str:
    prefix, name = tag.split(":")
    return f"{{{NS[prefix]}}}{name}"


def combined_text(sp: ET.Element) -> str:
    return " ".join(t.text or "" for t in sp.findall(".//a:t", NS))


def find_subtitle_shape(root: ET.Element, slide_no: int) -> ET.Element | None:
    shapes = root.findall(".//p:sp", NS)
    if slide_no == 1:
        for sp in shapes:
            if "Dynamic RT-DETR" in combined_text(sp):
                return sp
    for sp in shapes:
        text = combined_text(sp)
        if any(key in text for key in ("Rerun long-time", "Prepare three", "Design the", "Build a dashboard", "Ablation Study", "Connected a CSI")):
            return sp
    return None


def add_paragraph_to_shape(sp: ET.Element, text: str) -> None:
    tx_body = sp.find("p:txBody", NS)
    if tx_body is None:
        return
    body_pr = tx_body.find("a:bodyPr", NS)
    if body_pr is not None:
        body_pr.set("wrap", "square")
    p = ET.Element(qn("a:p"))
    p_pr = ET.SubElement(p, qn("a:pPr"))
    p_pr.set("marL", "0")
    p_pr.set("indent", "0")
    r = ET.SubElement(p, qn("a:r"))
    r_pr = ET.SubElement(r, qn("a:rPr"))
    r_pr.set("lang", "zh-CN")
    r_pr.set("sz", "1350")
    r_pr.set("b", "1")
    fill = ET.SubElement(r_pr, qn("a:solidFill"))
    ET.SubElement(fill, qn("a:srgbClr"), {"val": "1565C0"})
    ET.SubElement(r_pr, qn("a:latin"), {"typeface": "Microsoft YaHei"})
    ET.SubElement(r_pr, qn("a:ea"), {"typeface": "Microsoft YaHei"})
    t = ET.SubElement(r, qn("a:t"))
    t.text = text
    tx_body.append(p)

    xfrm = sp.find(".//a:xfrm", NS)
    if xfrm is not None:
        ext = xfrm.find("a:ext", NS)
        if ext is not None:
            old_cy = int(ext.get("cy", "0"))
            ext.set("cy", str(max(old_cy, 900000)))


def next_shape_id(root: ET.Element) -> int:
    ids = []
    for c_nv_pr in root.findall(".//p:cNvPr", NS):
        try:
            ids.append(int(c_nv_pr.get("id", "0")))
        except ValueError:
            pass
    return max(ids, default=1) + 1


def add_textbox(root: ET.Element, slide_no: int, text: str) -> None:
    sp_tree = root.find(".//p:cSld/p:spTree", NS)
    if sp_tree is None:
        return

    shape_id = next_shape_id(root)
    sp = ET.Element(qn("p:sp"))
    nv_sp_pr = ET.SubElement(sp, qn("p:nvSpPr"))
    ET.SubElement(nv_sp_pr, qn("p:cNvPr"), {"id": str(shape_id), "name": f"Takeaway {slide_no}"})
    ET.SubElement(nv_sp_pr, qn("p:cNvSpPr"), {"txBox": "1"})
    ET.SubElement(nv_sp_pr, qn("p:nvPr"))

    sp_pr = ET.SubElement(sp, qn("p:spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a:xfrm"))
    if slide_no == 12:
        x, y, cx, cy = int(0.7 * EMU), int(0.95 * EMU), int(11.9 * EMU), int(0.32 * EMU)
    else:
        x, y, cx, cy = int(3.6 * EMU), int(4.05 * EMU), int(8.4 * EMU), int(0.45 * EMU)
    ET.SubElement(xfrm, qn("a:off"), {"x": str(x), "y": str(y)})
    ET.SubElement(xfrm, qn("a:ext"), {"cx": str(cx), "cy": str(cy)})
    prst = ET.SubElement(sp_pr, qn("a:prstGeom"), {"prst": "roundRect"})
    ET.SubElement(prst, qn("a:avLst"))
    fill = ET.SubElement(sp_pr, qn("a:solidFill"))
    clr = ET.SubElement(fill, qn("a:srgbClr"), {"val": "F5F9FF"})
    ET.SubElement(clr, qn("a:alpha"), {"val": "92000"})
    ln = ET.SubElement(sp_pr, qn("a:ln"), {"w": "12700"})
    ln_fill = ET.SubElement(ln, qn("a:solidFill"))
    ET.SubElement(ln_fill, qn("a:srgbClr"), {"val": "1565C0"})

    tx_body = ET.SubElement(sp, qn("p:txBody"))
    ET.SubElement(
        tx_body,
        qn("a:bodyPr"),
        {"wrap": "square", "lIns": "91440", "tIns": "45720", "rIns": "91440", "bIns": "45720"},
    )
    ET.SubElement(tx_body, qn("a:lstStyle"))
    p = ET.SubElement(tx_body, qn("a:p"))
    r = ET.SubElement(p, qn("a:r"))
    r_pr = ET.SubElement(r, qn("a:rPr"), {"lang": "zh-CN", "sz": "1350", "b": "1"})
    r_fill = ET.SubElement(r_pr, qn("a:solidFill"))
    ET.SubElement(r_fill, qn("a:srgbClr"), {"val": "1565C0"})
    ET.SubElement(r_pr, qn("a:latin"), {"typeface": "Microsoft YaHei"})
    ET.SubElement(r_pr, qn("a:ea"), {"typeface": "Microsoft YaHei"})
    t = ET.SubElement(r, qn("a:t"))
    t.text = text
    sp_tree.append(sp)


def process_slide(data: bytes, slide_no: int) -> bytes:
    root = ET.fromstring(data)
    text = TAKEAWAYS[slide_no]
    if slide_no in APPEND_SLIDES:
        sp = find_subtitle_shape(root, slide_no)
        if sp is not None:
            add_paragraph_to_shape(sp, text)
        else:
            add_textbox(root, slide_no, text)
    elif slide_no in TEXTBOX_SLIDES:
        add_textbox(root, slide_no, text)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    slide_re = re.compile(r"ppt/slides/slide(\d+)\.xml$")
    with zipfile.ZipFile(PPTX_IN, "r") as zin, zipfile.ZipFile(PPTX_OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            match = slide_re.match(item.filename)
            if match:
                slide_no = int(match.group(1))
                if slide_no in TAKEAWAYS:
                    data = process_slide(data, slide_no)
            new_item = copy.copy(item)
            new_item.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(new_item, data)


if __name__ == "__main__":
    main()

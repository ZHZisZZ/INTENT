import * as d3 from "d3";
import * as dagreD3 from "dagre-d3";
import $ from "jquery";
import "../styles/graph.css";
import "../styles/table.css";
import { useEffect, useState, useContext } from "react";
import { BaselineContext } from "./Dashboard";
import { Typography } from "@material-ui/core";
import * as React from "react";
const cellHeight = 26;
const cellWidth = 20;
const fontSize = 13;
const bracketHeight = 26;
const bracketWidth = 5;
const bracketCell = (ch) =>
  `<td width="${bracketWidth}" height="${bracketHeight}" style=" font-size:${fontSize}px;" >${ch}</td>`;
const elementCell = (nodeName, elementID, element) => {
  return `<td class="cell" width="${cellWidth}" height="${cellHeight}" style="font-size:${fontSize}px; text-align:center" id="${nodeName}-${elementID}">${element}</td>`;
};
function getOffset(obj) {
  var childPos = $(`#${obj}`).offset();

  var parentPos = $("#dataflow").offset();

  var childOffset = {
    top: childPos.top - parentPos.top,
    left: childPos.left - parentPos.left,
  };
  return childOffset;
}
function getOffsetInner(obj) {
  var childPos = $(`#${obj}`).offset();
  var parentPos = $(".inner").offset();
  var childOffset = {
    top: childPos.top - parentPos.top,
    left: childPos.left - parentPos.left,
  };
  return childOffset;
}
const handleOutsideClick = (e) => {
  e.preventDefault();
  $(".custom-menu").hide();
  if (e.target && e.target.className !== "cell") {
    $(".provenance-edge").hide();
    document.removeEventListener("click", handleOutsideClick, false);
  }
};
const mockGraph = {
  edges: [
    {
      end: "n0",
      start: "p0",
    },
    {
      end: "p0",
      start: "n1",
    },
  ],
  nodes: {
    n0: {
      label: "tf.transpose(input_1)",
      type: "intermediate",
      value: "[[1, 4, 7], [2, 5, 8], [3, 6, 9]]",
    },
    n1: {
      label: "input_1",
      type: "input",
      value: "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
    },
    p0: {
      docstring: "Transpose `a`, where `a` is a Tensor.",
      value: "tf.transpose(a)",
    },
  },
  trace_edges: {
    n0: [
      {
        description: "",
        is_value_wise: false,
        provenance: "[[0], [3], [6], [1], [4], [7], [2], [5], [8]]",
        start: "n1",
      },
    ],
  },
};
function updatePreference(label, desired, preference, setPreference) {
  const newPreference = { ...preference };
  newPreference[label] = {
    label,
    desired,
  };
  setPreference(newPreference);
}
function regexIndexOf(string, regex, startpos) {
  var indexOf = string.substring(startpos || 0).search(regex);
  return indexOf >= 0 ? indexOf + (startpos || 0) : indexOf;
}
function recursiveFindArray(array, info, depth, maxDepth, isTuple = false) {
  let htmlStr = "";
  if (!Array.isArray(array)) {
    htmlStr += `<td class="cell" width="${cellWidth}" height="${cellHeight}" style="font-size:${fontSize}px;" id="${info.nodeName}-${info.elementIDStart}">${array}</td>`;
    info.elementIDStart++;
    return htmlStr;
  }
  if (!Array.isArray(array[0])) {
    // add the 1D array to the html
    htmlStr += bracketCell("[");
    array.forEach((element, index) => {
      if (index === array.length - 1) {
        htmlStr += elementCell(
          info.nodeName,
          info.elementIDStart + index,
          element
        );
      } else {
        htmlStr += elementCell(
          info.nodeName,
          info.elementIDStart + index,
          element + ","
        );
      }
    });
    info.elementIDStart += array.length;
    htmlStr += bracketCell("]");
  } else {
    // handle tuple
    if (isTuple && depth === 0) htmlStr += bracketCell("(");
    else htmlStr += bracketCell("[");
    array.forEach((element, index) => {
      // for each element on this dimension, recursively find the next dimension
      if (index > 0) {
        htmlStr += `<tr>`;
        for (let i = 0; i <= depth; i++) htmlStr += bracketCell(" ");
      }
      htmlStr += recursiveFindArray(element, info, depth + 1, maxDepth);

      if (index !== array.length - 1) {
        // htmlStr += `</tr>`;
        htmlStr += `<tr><td width="1" height="${Math.pow(
          maxDepth - depth - 1,
          2
        )}"> </td></tr>`;
      }
    });
    if (isTuple && depth === 0) htmlStr += bracketCell(")");
    else htmlStr += bracketCell("]");
  }
  return htmlStr;
}
function generateNodeLabelUpdate(node, nodeName) {
  var html = `<table class="table text-center">`;
  const {value, type, label} = node;
  let string = value;
  console.log(string);
  string = string.replaceAll('True', '"True"').replaceAll('False', '"False"')
  string = string.replaceAll("[inf", '["Inf"')
  string = string.replaceAll(" inf", ' "Inf"')
  string = string.replaceAll("[-inf", '["-Inf"')
  string = string.replaceAll(" -inf", ' "-Inf"')
  string = string.replaceAll("nan", '"nan"')
  let info = {
    nodeName,
    elementIDStart: 0,
  };
  let array = [];
  let maxDepth = 0;
  let isTuple = false;
  if(type === "input") html += `<div style="text-align: center">${label}</div>`
  if (string[0] === "[") {
    array = JSON.parse(string)
    while (string[maxDepth] === "[") maxDepth++;
  } else if (string[0] === "(") {
    let repString = string.replace("(", "[").replace(")", "]");
    try {
      array = JSON.parse(repString);
      isTuple = true;
    }
    catch(e){
      isTuple = false;
      array = string;
    }
  } else array = string;
  html += recursiveFindArray(array, info, 0, maxDepth, isTuple) + `</table>`;
  return html;
}
function generateNodeLabel(string, nodeName) {
  var html = `<table class="table text-center">`;
  var elementID = 0;

  if (string.indexOf("[") !== -1) {
    var open = 0;
    var close = 0;
    // Loop to check all occurrences
    while (true) {
      open = regexIndexOf(string, /\[[^\[]/gm, close);
      if (open !== -1) {
        html += `<tr>`;
        html += `<td width="15" height="26" style="padding:2px; font-size:13px;" >${string.substring(
          close,
          open + 1
        )}</td>`;
        close = regexIndexOf(string, /[^\]]\]/gm, open) + 1;
        const line = string.substring(open + 1, close);
        line
          .replaceAll(" ", "")
          .replaceAll(",", ",#")
          .split("#")
          .map((value) => {
            html += `<td class="cell" width="30" height="26" style="padding:2px; font-size:13px;" id="${nodeName}-${elementID}">${value}</td>`;
            elementID++;
          });
        const comma = string.indexOf(",", close);
        if (comma === -1) {
          html += `<td width="15" height="26" style="padding:2px; font-size:13px;" >${string.substring(
            close
          )}</td>`;
        } else {
          html += `<td width="15" height="26" style="padding:2px; font-size:13px;" >${string.substring(
            close,
            comma + 1
          )}</td>`;
          close = comma + 1;
        }
        html += `</tr>`;
      } else {
        break;
      }
    }
  } else {
    html += `<tr>`;
    if (string.indexOf("(") !== -1) {
      html += `<td width="15" height="26" style="padding:2px; font-size:13px;" >${string.substring(
        0,
        1
      )}</td>`;
      const close = string.indexOf(")");
      const line = string.substring(1, close);
      line
        .replaceAll(" ", "")
        .replaceAll(",", ",#")
        .split("#")
        .map((value) => {
          html += `<td class="cell" width="30" height="26" style="padding:2px; font-size:13px;" id="${nodeName}-${elementID}">${value}</td>`;
          elementID++;
        });
      html += `<td width="15" height="26" style=" padding:2px; font-size:13px;" >${string.substring(
        close
      )}</td>`;
    } else {
      html += `<td class="cell" width="30" height="26" style="padding:2px; font-size:13px;" id="${nodeName}-0">${string}</td>`;
    }
    html += `</tr>`;
  }

  html += `</table>`;
  return html;
}

function renderDataflow({
  graph,
  setCurrentNodeId,
  currentNodeId,
  preference,
  setPreference,
}) {
  // Cleanup old graph
  let template = graph;
  var svg = d3.select("#dataflow");
  svg.selectAll("*").remove();

  if (!template) {
    template = mockGraph;
    return <div></div>;
  }
  // Create the input graph
  var g = new dagreD3.graphlib.Graph({ multigraph: true }).setGraph({});

  const nodes = template["nodes"];
  const edges = template["edges"];
  const trace = template["trace_edges"];

  for (const [key, value] of Object.entries(nodes)) {
    if (key[0] === "n") {
      g.setNode(key, {
        labelType: "html",
        label: generateNodeLabelUpdate(value, key),
        style: "fill:#fff",
        padding: 0,
      });
    } else {
      let color = "rgba(102, 178, 255, 0.15)";
      if (preference[value["value"]] !== undefined)
        color = preference[value["value"]].desired ? "#4caf5026" : "#f4433626";
      g.setNode(key, {
        label: value["value"],
        style: `fill:${color}`,
        shape: "ellipse",
      });
    }
  }

  for (const [key, value] of Object.entries(edges)) {
    g.setEdge(value["start"], value["end"], {
      label: value["label"],
      curve: d3.curveBasis,
    });
  }

  // Create the renderer
  var render = new dagreD3.render();

  // Set up an SVG group so that we can translate the final graph.
  var inner = svg.append("g").attr("class", "inner");
    // svg
    // .call(
    //   d3.zoom().on("zoom", function (event) {
    //     inner.attr("transform", event.transform);
    //   }))
    // .on("dblclick.zoom", null)
    // .on("wheel.zoom", null);
  // Run the renderer. This is what draws the final graph.
  render(inner, g);

  // Center the graph
  svg.attr("height", g.graph().height + 200);
  svg.attr("width", "100%");
  inner.attr("transform", "translate(" + 100 + ", 20)");
  // doc string hover
  svg
    .selectAll("g.node")
    .on("mouseover", function (event, id) {
      if (id[0] === "p") {
        // && !isDown[id]) {
        $("#operation-doc").html(nodes[id]["docstring"]);
        $(".op-description")
          .finish()
          .toggle()
          // In the right position (the mouse)
          .css({
            top: event.pageY + "px",
            left: event.pageX + "px",
          });
      }
    })
    .on("mouseleave", (event, id) => {
      if (id[0] === "p") {
        // && !isDown[id]) {
        $(".op-description").hide();
      }
    });
  // text element also trigger doc, not just the shape
  svg
    .selectAll("g.node .label")
    .on("mouseover", function (event, id) {
      if (id[0] === "p") {
        // && !isDown[id]) {
        $("#operation-doc").html(nodes[id]["docstring"]);
        $(".op-description")
          .finish()
          .toggle()
          // In the right position (the mouse)
          .css({
            top: event.pageY + "px",
            left: event.pageX + "px",
          });
      }
    })
    .on("mouseleave", (event, id) => {
      if (id[0] === "p") {
        // && !isDown[id]) {
        $(".op-description").hide();
      }
    });
  svg.selectAll("g.node").on("contextmenu", async function (event, id) {
    if (event.button === 2) {
      var id = $(this)[0].__data__;
      if (id[0] === "p") {
        event.preventDefault();
        $(".op-description").hide(100);
        $(".custom-menu")
          .finish()
          .toggle()
          // In the right position (the mouse)
          .css({
            top: event.pageY + "px",
            left: event.pageX + "px",
          });
        document.addEventListener("click", handleOutsideClick, false);
        await setCurrentNodeId(id);
      }
    }
  });
  d3.selectAll(".custom-menu li").on("click", function (e) {
    // This is the triggered action name
    updatePreference(
      nodes[currentNodeId].value,
      $(this).attr("data-action") === "desired",
      preference,
      setPreference
    );
    $(".custom-menu").hide(100);
  });
  const recursiveFindProvenance = (secondary, celid ) =>{
    let node_id = celid.split("-")[0],
      cell_num = celid.split("-")[1];
    if (node_id in trace) {
      const axis_x = getOffsetInner(celid).left,
        axis_y = getOffsetInner(celid).top;
      for (const [key, value] of Object.entries(trace[node_id])) {
        if (value["provenance"] === "None") continue;
        let provenance = eval(value["provenance"]),
          next = value["start"];
        let id_list = provenance[parseInt(cell_num)];
        for (const index of id_list) {
          let next_id = next + "-" + index.toString();
          const next_x = getOffsetInner(next_id).left,
            next_y = getOffsetInner(next_id).top;
          let path = d3.path();
          path.moveTo(axis_x + 15, axis_y + 10);
          path.lineTo(next_x + 15, next_y + 16);
          // Set up an SVG group so that we can translate the final graph.
          const opacity = (secondary) ? 0.2 : 0.4;
          const stroke = "4,4";
          inner
            .append("path")
            .attr("d", path)
            .attr("stroke", "red")
            .attr("style", `opacity:${opacity}`)
            .style("stroke-dasharray", stroke)
            .attr("class", "provenance-edge")
            .attr("stroke-width", "4");
          recursiveFindProvenance(true, next_id);
        }
      }
    }
  }
  d3.selectAll(".cell").on("click", function () {
    $(".provenance-edge").remove();
    let celid = d3.select(this).attr("id");
    recursiveFindProvenance(false, celid)
    document.addEventListener("click", handleOutsideClick, false);
  });
}
const Graph = ({
  allOutputTensors,
  validations,
  selectedTestCase,
  selectedSolution,
  preference,
  setPreference,
}) => {
  const baseline = useContext(BaselineContext);
  const [currentNodeId, setCurrentNodeId] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  useEffect(() => {
    if (!baseline) {
      const data =
        validations["example" + selectedTestCase]?.[
          "solution" + selectedSolution
        ];
      if (!data || data.exception) {
        const svg = d3.select("#dataflow");
        svg.selectAll("*").remove();
        svg.attr("height", 0);
        svg.attr("width", 0);
        if (data?.exception) {
          setErrorMessage("Exception: " + data.eval_output);
        } else {
          setErrorMessage("");
        }
      } else if (!data.match) {
        const expectedTensor = allOutputTensors[
          selectedTestCase
        ].stringValue.replaceAll(" ", "");
        const actualTensor = data.eval_output.replaceAll(" ", "");
        setErrorMessage(
          `Expected tensor ${expectedTensor} but got ${actualTensor}`
        );
        renderDataflow({
          graph: data.graphs,
          currentNodeId,
          setCurrentNodeId,
          preference,
          setPreference,
        });
      } else {
        setErrorMessage("");
        renderDataflow({
          graph: data.graphs,
          currentNodeId,
          setCurrentNodeId,
          preference,
          setPreference,
        });
      }
    }
  }, [
    validations,
    selectedTestCase,
    selectedSolution,
    preference,
    currentNodeId,
  ]);
  return (
    <div style={{height: "100%"}}>
      <Typography
        style={{
          fontSize: 14,
        }}
        color="error"
        gutterBottom
      >
        {errorMessage}
      </Typography>
      <svg id="dataflow" />
      <div className="node-op">
        <ul className="custom-menu">
          <li data-action="desired">Add to desired</li>
          <li data-action="undesired">Add to undesired</li>
        </ul>
      </div>
      <div className="op-des">
        <ul className="op-description">
          <li id="operation-doc" />
        </ul>
      </div>
    </div>
  );
};

export default Graph;

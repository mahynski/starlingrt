"""
Visualize GCMS data to determine consensus and any corrections necessary.
Author: Nathan A. Mahynski

"""
import numpy as np
from starlingrt import functions, data

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import layout
from bokeh.models import (ColumnDataSource, DataTable, HoverTool, CrosshairTool, FileInput, # type: ignore [attr-defined]
                          SelectEditor, CustomJS, Segment, VBar, Rect, Button, TextInput, RadioButtonGroup, Span, # type: ignore [attr-defined]
                          StringFormatter, TableColumn, RangeSlider, Slider, Select, CDSView, IndexFilter, HTMLTemplateFormatter) # type: ignore [attr-defined]

def make(top_entries: dict[str, data.Entry], width: int, threshold: float, output_filename: str = "summary.html") -> None:
    """
    Make the interactive HTML document for users to inspect.

    Parameters
    ----------
    top_entries : dict(str, Entry)
        Top entries (hit number = 1) labeled by their sha1 hash.

    width : int
        Width of the HTML table output.

    threshold : float
        Minimum retention time gap between consecutive compounds to be resolved as different.

    output_filename : str, optional(default="summary.html")
        Name of HTML file to save results to.
    """
    # Get Data and perform initial calculations in python
    df, name_groups, ordered_cats = functions.get_dataframe(top_entries)
    df_iqr = functions.get_quantiles_df(name_groups)
    by_name = data.Utilities.group_entries_by_name(top_entries)

    # Create column for changeable name
    rt_groups = functions.group_by_rt_step(df, threshold)
    suggested_name, ties, entropy = functions.suggest_names(rt_groups)
    df = functions.assign_suggestions(df, rt_groups, suggested_name, ties)
    df['new_name'] = df['suggested_name'].copy()

    # Create sources for Bokeh JavaScript
    total_source = ColumnDataSource(df)
    iqr_source = ColumnDataSource(df_iqr)
    view = CDSView(filter=IndexFilter(np.arange(len(df)).tolist()))

    # Make Table
    compound_names = sorted(df["hit_name"].unique()) + ['unknown'] # Lowercase 'unknown' for easier autocomplete
    filenames = sorted(df["origin"].unique())

    # https://stackoverflow.com/questions/50996875/how-to-color-rows-and-or-cells-in-a-bokeh-datatable
    template="""
            <div style="background:<%= 
                (function colorfromint(){
                    if (flag === "" && hit_name === suggested_name)
                        {return("green")}
                    else if (flag === "")
                        {return("yellow")}
                    else
                        {return("red")}
                    }()) %>; 
                color: black"> 
            <%= value %>
            </div>
            """
    formatter =  HTMLTemplateFormatter(template=template)

    columns = [
        TableColumn(field="rt", title="Retention Time",
                    ),
        TableColumn(field="quality", title="Quality",
                    ),
        TableColumn(field="hit_name", title="Original Name", # Original name is not changeable
                    formatter=StringFormatter(font_style="bold")),
        TableColumn(field="suggested_name", title="Suggested Name", # Computer suggested name is not changeable
                    formatter=StringFormatter(font_style="bold")),
        TableColumn(field="second_suggestion", title="Second Suggestion", # Computer suggested name is not changeable
                    formatter=StringFormatter(font_style="bold")),
        TableColumn(field="new_name", title="Assigned Name", 
                    editor=SelectEditor(options=compound_names), # Can be modified based on user choice
                    formatter=formatter),
        TableColumn(field="hit_number", title="Hit Number",
                    formatter=StringFormatter(font_style="bold")),
        TableColumn(field="peak_width", title="Peak Width",
                    ),
        TableColumn(field="area", title="Area",
                    ),
        TableColumn(field="origin", title="Filename",
                    ),
        TableColumn(field="flag", title="Flag",
                    ),
        TableColumn(field="hash", title="Entry Hash",
                    ),
    ]

    data_table = DataTable(source=total_source,
                        view=view,
                       columns=columns, # Based on all possibilities regardless of quality
                       editable=True, #False 
                       width=width,
                       index_position=-1, index_header="Index", index_width=60)

    # Make Figure
    p = figure(
        background_fill_color="#efefef", 
        x_range=ordered_cats, 
        title="Retention Time Ranges",
        width=width,
        height=int(width*0.6),
        tools="pan,wheel_zoom,ybox_select,xbox_select,lasso_select,reset", 
        active_drag="pan",
        x_axis_label="",
        y_axis_label="Retention Time",
    )

    # Whisker plot
    # Stems
    upper_seg = Segment(x0="new_name", y0="upper", x1="new_name", y1="q3", line_color="black", line_alpha=0.6)
    lower_seg = Segment(x0="new_name", y0="lower", x1="new_name", y1="q1", line_color="black", line_alpha=0.6)
    p.add_glyph(iqr_source, upper_seg)
    p.add_glyph(iqr_source, lower_seg)

    # Boxes
    upper_bar = VBar(x="new_name", width=0.7, bottom="q2", top="q3", fill_color="#E08E79", line_color="black", fill_alpha=0.6, line_alpha=0.6)
    lower_bar = VBar(x="new_name", width=0.7, bottom="q1", top="q2", fill_color="#3B8686", line_color="black", fill_alpha=0.6, line_alpha=0.6)
    p.add_glyph(iqr_source, upper_bar)
    p.add_glyph(iqr_source, lower_bar)

    # Whiskers (almost-0 height rects simpler than segments)
    upper_whisker = Rect(x="new_name", y="upper", width=0.2, height=0.0001, line_color="black", fill_color='black', fill_alpha=0.6, line_alpha=0.6)
    lower_whisker = Rect(x="new_name", y="lower", width=0.2, height=0.0001, line_color="black", fill_color='black', fill_alpha=0.6, line_alpha=0.6)
    p.add_glyph(iqr_source, upper_whisker)
    p.add_glyph(iqr_source, lower_whisker)

    # Build IQR inspection plots
    k_step = 0.5
    k_values = np.arange(1.0, 3.51, k_step).tolist()

    change_hist_code = """
    var name = compound_select.value;
    var k = k_select.value.toString();

    var new_data = {};
    new_data['top'] = compounds_hist[name];
    new_data['bottom'] = Array(compounds_hist[name].length).fill(0);
    new_data['right'] = compounds_edges[name].slice(1);
    new_data['left'] = compounds_edges[name].slice(0, -1);

    source.data = new_data;
    source.change.emit();

    var new_dict_c = {}
    new_dict_c['x'] = points[name][k]['concern']['x'];
    new_dict_c['y'] = points[name][k]['concern']['y'];
    concerns.data = new_dict_c;

    var new_dict_nc = {}
    new_dict_nc['x'] = points[name][k]['not_concern']['x'];
    new_dict_nc['y'] = points[name][k]['not_concern']['y'];
    not_concern.data = new_dict_nc;

    concerns.change.emit();
    not_concern.change.emit();

    var curr_data = {};
    curr_data['x'] = [];
    curr_data['y'] = [];
    curr_data['Index'] = [];
    for (let i = 0; i < view.filter.indices.length; ++i) {
        let idx = view.filter.indices[i];
        if (total_source.data.new_name[idx] === name) {
            curr_data['x'].push(total_source.data.rt[idx]);
            curr_data['y'].push(curr_data['x'].length);
            curr_data['Index'].push(idx); 
        }
    }
    source_current.data = curr_data;

    source_current.change.emit();
    """

    compounds_hist, compounds_edges, points = functions.make_histograms(by_name, k_values, bins=10)

    def compare_factory(dropdown=True):
        compare = figure(
            y_axis_label="Counts",
            x_axis_label="Retention Time",
        )
        
        name = list(by_name.keys())[0]

        source = ColumnDataSource(
            dict(
                left=compounds_edges[name][:-1],
                right=compounds_edges[name][1:],
                top=compounds_hist[name],
                bottom=[0]*len(compounds_hist[name]),
            )
        )

        source_concerns = ColumnDataSource(dict(
            x=points[name][str(np.min(k_values))]['concern']['x'],
            y=points[name][str(np.min(k_values))]['concern']['y']
        ))

        source_not_concern = ColumnDataSource(dict(
            x=points[name][str(np.min(k_values))]['not_concern']['x'],
            y=points[name][str(np.min(k_values))]['not_concern']['y']
        ))

        current = {'x':[], 'y':[], 'Index':[]}
        for idx in view.filter.indices:
            if (total_source.data['new_name'][idx] == name):
                current['x'].append(total_source.data['rt'][idx])
                current['y'].append(len(current['x']))
                current['Index'].append(idx)
        source_current = ColumnDataSource(dict(
            x=current['x'],
            y=current['y'],
            Index=current['Index']
        )) 

        compare.quad(
            source=source,
            fill_color="navy", line_color="white", alpha=0.5)

        compare.circle(
            x="x", y="y", color="red", 
            size=10, alpha=0.5, 
            source=source_concerns, legend_label='(Original) Concerning'
        )

        compare.circle(
            x="x", y="y", color="blue", 
            size=10, alpha=0.5, 
            source=source_not_concern, legend_label='(Original) Not Concerning'
        )

        current_points = compare.star(
            x="x", y="y", color="black",
            size=10, alpha=0.5,
            source=source_current, legend_label='(Visible) Currently Assigned'
        )

        compare.legend.location = "top_left"
        compare.legend.click_policy = "hide"
        
        if dropdown: # Downdown menu for compounds
            compound_select = Select(title="Select Compound", value=name, 
            options=[name for name, entries in sorted(by_name.items(), key=lambda x:np.median([e[0].rt for e in x[1]]))]) # Sort by RT
        else: # Manually enter
            compound_select = TextInput(title="Select Compound", value=name)

        k_select = Select(title="K Value", value=str(np.min(k_values)), options=[str(x) for x in k_values])
        
        callback = CustomJS(
            args=dict(
                source=source, 
                concerns=source_concerns,
                not_concern=source_not_concern,
                source_current=source_current,
                view=view,
                total_source=total_source,
                compounds_hist=compounds_hist, 
                compounds_edges=compounds_edges,
                points=points,
                compound_select=compound_select,
                k_select=k_select
            ),
            code=change_hist_code
        )
        
        compound_select.js_on_change(
            "value", 
            callback
        )
        
        k_select.js_on_change(
            "value", 
            callback
        )

        htool = HoverTool(renderers=[current_points], tooltips=[("RT", "@x"), ("Index", "@Index")])
        compare.add_tools(htool)
        
        return compare, source, source_concerns, source_not_concern, source_current, compound_select, k_select

    compare_1, source_1, source_concerns_1, source_not_concern_1, source_current_1, compound_select_1, k_select_1 = compare_factory(dropdown=True)
    compare_2, source_2, source_concerns_2, source_not_concern_2, source_current_2, compound_select_2, k_select_2 = compare_factory(dropdown=True)
    compare_3, source_3, source_concerns_3, source_not_concern_3, source_current_3, compound_select_3, k_select_3 = compare_factory(dropdown=True)

    # Link plots with crosshair
    i_width = Span(dimension="width", line_dash="dashed", line_width=2)
    i_height = Span(dimension="height", line_dash="dotted", line_width=2)
    compare_1.add_tools(CrosshairTool(overlay=[i_width, i_height]))
    compare_2.add_tools(CrosshairTool(overlay=[i_width, i_height]))
    compare_3.add_tools(CrosshairTool(overlay=[i_width, i_height]))

    alt_compare_1, alt_source_1, alt_source_concerns_1, alt_source_not_concern_1, alt_source_current_1, alt_compound_select_1, alt_k_select_1 = compare_factory(dropdown=False)
    alt_compare_2, alt_source_2, alt_source_concerns_2, alt_source_not_concern_2, alt_source_current_2, alt_compound_select_2, alt_k_select_2 = compare_factory(dropdown=False)
    alt_compare_3, alt_source_3, alt_source_concerns_3, alt_source_not_concern_3, alt_source_current_3, alt_compound_select_3, alt_k_select_3 = compare_factory(dropdown=False)

    # Link plots with crosshair
    alt_i_width = Span(dimension="width", line_dash="dashed", line_width=2)
    alt_i_height = Span(dimension="height", line_dash="dotted", line_width=2)
    alt_compare_1.add_tools(CrosshairTool(overlay=[alt_i_width, alt_i_height]))
    alt_compare_2.add_tools(CrosshairTool(overlay=[alt_i_width, alt_i_height]))
    alt_compare_3.add_tools(CrosshairTool(overlay=[alt_i_width, alt_i_height]))

    # Min observation slider (filter)
    min_obs_slider = Slider(start=0, 
                            end=name_groups['hit_name'].count().max(), # Based on original assignment
                            value=0, 
                            step=1, 
                            title="Minimum Observations within Quality Range")

    # Minimum Quality slider (filter)
    quality_slider = RangeSlider(start=0, end=100, step=1, value=(1, 99), title="Quality Range")

    # This computes all the statistics, summaries, etc. to display in various glyphs.  This is all in one
    # function so that different events, when triggered, will all call the same function and update
    # everything together to remain in sync.
    recompute_code = """
        var qvals = quality_slider.value;
        
        view.filter.indices = [];

        // Get the data from the data sources
        var t = total_source.data
        var iqr_ = iqr_source.data;
        var iqr_data = {};
        
        iqr_.new_name = []
        iqr_.q1 = []
        iqr_.q2 = []
        iqr_.q3 = []
        iqr_.upper = []
        iqr_.lower = []

        var counts = {};
        for (var i = 0; i < t.index.length; i++) {  
            counts[t.new_name[i]] = 0;
        }
        for (var i = 0; i < t.index.length; i++) {  
            counts[t.new_name[i]] += 1;
        }

        // Initialize IQR data
        for (var i = 0; i < t.index.length; i++) { 
            if ((qvals[0] <= t.quality[i]) && (t.quality[i] <= qvals[1]) && (counts[t.new_name[i]] > min_obs_slider.value)) {
                iqr_data[t.new_name[i]] = [];
            }
        }

        var data_1 = {};
        data_1['x'] = [];
        data_1['y'] = [];
        data_1['Index'] = [];
        var data_2 = {};
        data_2['x'] = [];
        data_2['y'] = [];
        data_2['Index'] = [];
        var data_3 = {};
        data_3['x'] = [];
        data_3['y'] = [];
        data_3['Index'] = [];

        // Update the visible data
        for (var i = 0; i < t.index.length; i++) { 
            if ((qvals[0] <= t.quality[i]) && (t.quality[i] <= qvals[1]) && (counts[t.new_name[i]] > min_obs_slider.value)) {
                view.filter.indices.push(i);
                
                // Accumulate rentention times
                iqr_data[t.new_name[i]].push(t.rt[i]);

                if (t.new_name[i] === compound_select_1.value) {
                    data_1['x'].push(t.rt[i]);
                    data_1['y'].push(data_1['x'].length);
                    data_1['Index'].push(t.index[i]);
                }
                if (t.new_name[i] === compound_select_2.value) {
                    data_2['x'].push(t.rt[i]);
                    data_2['y'].push(data_2['x'].length);
                    data_2['Index'].push(t.index[i]);
                }
                if (t.new_name[i] === compound_select_3.value) {
                    data_3['x'].push(t.rt[i]);
                    data_3['y'].push(data_3['x'].length);
                    data_3['Index'].push(t.index[i]);
                }
            }   
        }
        source_current_1.data = data_1;
        source_current_2.data = data_2;
        source_current_3.data = data_3;

        view.change.emit();
        source_current_1.change.emit();
        source_current_2.change.emit();
        source_current_3.change.emit();
        
        // Update IQR visuals
        // From: https://www.geeksforgeeks.org/interquartile-range-iqr/
        function QUANTILE(data, q)
        {
            // R-7 method: https://en.wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
            let values = data.slice().sort((a, b) => a - b); // copy and sort
            let h = (data.length - 1)*q;

            if (h % 1 === 0) {
                return values[h]
            } else {
                let base = Math.floor(h);
                return values[base] + (h - base)*(values[Math.ceil(h)] - values[base]);
            }
        }
        
        var iqr_values = [0, 0, 0];
        for (const [new_name, values] of Object.entries(iqr_data)) {
            iqr_values = [QUANTILE(values, 0.25), QUANTILE(values, 0.5), QUANTILE(values, 0.75)];
            iqr_.new_name.push(new_name);
            iqr_.q1.push(iqr_values[0]);
            iqr_.q2.push(iqr_values[1]);
            iqr_.q3.push(iqr_values[2]);
            iqr_.lower.push(iqr_values[0] - 1.5*(iqr_values[2] - iqr_values[0]));
            iqr_.upper.push(iqr_values[2] + 1.5*(iqr_values[2] - iqr_values[0]));
        }
        iqr_source.change.emit();

        total_source.change.emit();
    """

    recompute = CustomJS(args=dict(
        total_source = total_source, 
        view = view, 
        iqr_source = iqr_source, 
        min_obs_slider = min_obs_slider, 
        quality_slider = quality_slider,
        source_current_1 = source_current_1, 
        compound_select_1 = compound_select_1,
        source_current_2 = source_current_2, 
        compound_select_2 = compound_select_2,
        source_current_3 = source_current_3, 
        compound_select_3 = compound_select_3,
        ), code=recompute_code)

    # Recompute should be triggered by filters (quality and min_obs) and user edits to the datatable
    quality_slider.js_on_change('value', recompute)
    min_obs_slider.js_on_change('value', recompute)
    total_source.js_on_change('patching', recompute)

    # Datapoints
    points = p.circle(x="new_name", y="rt", color="#F38630", size=4, alpha=0.5, 
                    source=total_source, 
                    view=view
                    )
    
    tooltips = [
        ("Original Name", "@hit_name"),
        ("Quality", "@quality"),
        ("File", "@origin"),
    ]

    hover_tool = HoverTool(renderers=[points], tooltips=tooltips)
    p.add_tools(hover_tool)

    crosshair_tool = CrosshairTool()
    p.add_tools(crosshair_tool)

    iqr_columns = [
            TableColumn(field="new_name", title="Name",
                formatter=StringFormatter(font_style="bold")),
            TableColumn(field="lower", title="Lower Whisker",
                ),
            TableColumn(field="q1", title="Q1",
                ),
            TableColumn(field="q2", title="Q2",
                ),
            TableColumn(field="q3", title="Q3",
                ),
            TableColumn(field="upper", title="Upper Whisker",
                ),
    ]

    iqr_table = DataTable(source=iqr_source,
            columns=iqr_columns, 
            editable=False,
            width=width,
            index_position=-1, index_header="Index", index_width=60)

    filename_input_iqr = TextInput(value="exported_iqr.txt", title="") #Export quartile data to:")
    download_button_iqr = Button(label='Export Tab Delimited File', button_type='success') 
    download_button_iqr.js_on_click(CustomJS( # https://github.com/surfaceowl-ai/python_visualizations/blob/main/notebooks/bokeh_save_export_data.py
        args=dict(iqr_source=iqr_source, filename_input=filename_input_iqr),
        code="""
            function table_to_csv(source) {
                const columns = ['new_name', 'lower', 'q1', 'q2', 'q3', 'upper'];
                const nrows = source.get_length();
                var lines = ["Name\tLower Whisker\tQ1\tQ2\tQ3\tUpper Whisker"]; 

                for (let i = 0; i < source.get_length(); i++) {
                    lines += '\\n';
                    for (let j = 0; j < columns.length; j++) {
                        lines += source.data[columns[j]][i].toString();
                        if (j < columns.length-1) {
                            lines += '\\t';
                        }
                    }
                }
                return lines;
            }

            var file = new Blob([table_to_csv(iqr_source)], {type: 'text/plain'});
            var elem = window.document.createElement('a');
            elem.href = window.URL.createObjectURL(file);
            elem.download = filename_input.value;
            document.body.appendChild(elem);
            elem.click();
            document.body.removeChild(elem);
            """,
        )
    )

    filename_input = TextInput(value="saved_state.bt", title="")#Export to (must have .bt suffix):")
    radio = RadioButtonGroup(labels=['Selected Points Only', 'Entire Table'], active=0)
    download_button = Button(label='Save Current State to Tab Delimited File', button_type='success') # CSV gets confused with commas in hit_names
    download_button.js_on_click(CustomJS( # https://github.com/surfaceowl-ai/python_visualizations/blob/main/notebooks/bokeh_save_export_data.py
        args=dict(total_source=total_source, filename_input=filename_input, radio=radio, min_obs_slider=min_obs_slider, quality_slider=quality_slider),
        code="""
            var inds;
            if (radio.active == 0) { // Only selected points
                inds = total_source.selected.indices;
            } else { // All points in the table
                inds = [];
                for (let i=0; i < total_source.get_length(); ++i) {
                    inds.push(i);
                }
            }

            function table_to_csv(source) {
                const columns = ['rt', 'quality', 'hit_name', 'suggested_name', 'new_name', 'hit_number', 'mol_weight', 'absolute_height', 'baseline_height', 'peak_width', 'area', 'origin', 'compound_number', 'library', 'entry_number_library', 'scan_number', 'flag', 'hash'];
                const nrows = source.get_length();
                var lines = 'Minimum Observations within Quality Range\t' + min_obs_slider.value.toString() + '\tQuality Range\t' + quality_slider.value[0].toString() + '\t' + quality_slider.value[1].toString();
                lines += ['\\nRetention Time\tQuality\tOriginal Name\tSuggested Name\tAssigned Name\tHit Number\tMolecular Weight\tAbsolute Height\tBaseline Height\tPeak Width\tArea\tFilename\tCompound Number\tLibrary\tLibrary Number\tScan Number\tFlag\tHash'];

                for (let i = 0; i < inds.length; i++) {
                    lines += '\\n';
                    for (let j = 0; j < columns.length; j++) {
                        lines += source.data[columns[j]][inds[i]].toString() 
                        if (j < columns.length-1) {
                            lines += '\\t';
                        }
                    }
                }
                return lines;
            }

            var file = new Blob([table_to_csv(total_source)], {type: 'text/plain'});
            var elem = window.document.createElement('a');
            elem.href = window.URL.createObjectURL(file);
            elem.download = filename_input.value;
            document.body.appendChild(elem);
            elem.click();
            document.body.removeChild(elem);
            """,
        )
    )

    filename_import = FileInput(accept=".bt,")
    import_button = Button(label="Load Previous State", button_type='danger')
    import_button.js_on_click(CustomJS( 
        args=dict(source=total_source, 
            filename_import=filename_import, quality_slider=quality_slider, min_obs_slider=min_obs_slider),
        code="""
            var data = atob(filename_import.value); // Convert from base64
            var lines = data.split('\\n');        

            // These edits will trigger new visible data
            min_obs_slider.value = parseInt(lines[0].split('\\t')[1]);
            quality_slider.value = [parseInt(lines[0].split('\\t')[3]), parseInt(lines[0].split('\\t')[4])];

            for (let i = 1; i < lines.length; ++i) {
                let row = lines[i].split('\\t');
                let hash = row[row.length - 1];
                let index = -1;

                // Try to find entry in total source and update it (based on hash)
                for (let j = 0; j < source.get_length(); ++j) {
                    if (hash === source.data.hash[j]) {
                        index = j;
                        break;
                    }
                }
                if (index >= 0) {
                    source.data.rt[index] = parseFloat(row[0]);
                    source.data.quality[index] = parseInt(row[1]);
                    source.data.hit_name[index] = row[2];
                    source.data.suggested_name[index] = row[3];
                    source.data.new_name[index] = row[4];
                    source.data.hit_number[index] = parseInt(row[5]);
                    source.data.mol_weight[index] = parseFloat(row[6]);
                    source.data.absolute_height[index] = parseInt(row[7]);
                    source.data.baseline_height[index] = parseInt(row[8]);
                    source.data.peak_width[index] = parseFloat(row[9]);
                    source.data.area[index] = parseInt(row[10]);
                    source.data.origin[index] = row[11];
                    source.data.compound_number[index] = parseInt(row[12]);
                    source.data.library[index] = row[13];
                    source.data.entry_number_library[index] = parseInt(row[14]);
                    source.data.scan_number[index] = parseInt(row[15]);
                    source.data.flag[index] = row[16]
                }
            }
            source.change.emit();

            """,
        )
    )
    import_button.js_on_click(recompute) # Also trigger a re-compute after loading the data

    p.xaxis.major_label_orientation = "vertical"
    output_file(filename=output_filename, title="Interactive Retention Time Summary")
    save(
        layout([
            [filename_import, import_button, filename_input, radio, download_button],
            [data_table], 
            [min_obs_slider, quality_slider],
            [p],
            [k_select_1, k_select_2, k_select_3], 
            [compound_select_1, compound_select_2, compound_select_3], 
            [compare_1, compare_2, compare_3],
            [alt_k_select_1, alt_k_select_2, alt_k_select_3], 
            [alt_compound_select_1, alt_compound_select_2, alt_compound_select_3],
            [alt_compare_1, alt_compare_2, alt_compare_3],
            [filename_input_iqr, download_button_iqr],
            [iqr_table]
        ], sizing_mode='scale_width')
    )


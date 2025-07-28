clear,clc
json2latex("conv1/conv1_1Month_point1H.json", "table_2.tex");
res_str = jsondecode(fileread("conv1/conv1_1Day_point1H.json"));
res_str_2 = jsondecode(fileread("conv1/conv1_1Month_point1H.json"));

%% helper functions
function json2latex(jsonFilename, texFilename)
% JSON2LATEX  Convert a JSON array of results into a LaTeX table.
%   json2latex('data.json','results.tex') reads data.json,
%   and writes a LaTeX tabular to results.tex.

  %--- 1. Read & parse JSON
  txt  = fileread(jsonFilename);
  data = jsondecode(txt);

  %--- 2. Extract unique nf values & sort
  nfs = unique([data.nf]);
  % sort ascending
  nfs = sort(nfs);

  %--- 3. Prepare output file
  fid = fopen(texFilename,'w');
  assert(fid~=-1, 'Could not open %s for writing.', texFilename);

  %--- 4. Write LaTeX header
  fprintf(fid, '\\begin{table}[tph]\n');
  fprintf(fid, '  \\centering\n');
  fprintf(fid, '  \\begin{tabular}{lccccc}\n');
  fprintf(fid, '    \\hline\\hline\n');
  fprintf(fid, '    $n$ & Price & RMSE & SE & Bias & CPU time (s) \\\\\n');
  fprintf(fid, '    \\hline\n');

  %--- 5. Loop over each nf block
  for i = 1:numel(nfs)
    this_nf = nfs(i);
    % select entries
    sel = data([data.nf] == this_nf);
    % sort by n
    [~, idx] = sort([sel.n]);
    sel = sel(idx);

    % print a subheading row for this nf
    fprintf(fid, '    \\multicolumn{6}{l}{$n_f = %d$, $\\text{Reference Price} = %.6f$} \\\\\n', this_nf, sel(1).ref);

    % print each row
    for j = 1:numel(sel)
      row = sel(j);
      fprintf(fid, '    %4d & %.6f & %.3e & %.3e & %.3e & %.2f \\\\\n', ...
              row.n, row.price, row.rmse, row.se, row.bias, row.t);
    end

    % blank line between blocks (but not after last)
    if i < numel(nfs)
      fprintf(fid, '    \\addlinespace\n');
    end
  end

  %--- 6. Write footer
  fprintf(fid, '    \\hline\n');
  fprintf(fid, '  \\end{tabular}\n');
  fprintf(fid, '  \\caption{Simulation results grouped by $n_f$.}\n');
  fprintf(fid, '  \\label{tab:sim_results}\n');
  fprintf(fid, '\\end{table}\n');

  fclose(fid);
  fprintf('Wrote LaTeX table to %s\n', texFilename);
end

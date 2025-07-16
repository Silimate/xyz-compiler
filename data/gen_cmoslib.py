print('''library(cmoslib) {
  capacitive_load_unit (1,fF);
  current_unit : "1mA";
  delay_model : "table_lookup";
  input_threshold_pct_fall : 50;
  input_threshold_pct_rise : 50;
  output_threshold_pct_fall : 50;
  output_threshold_pct_rise : 50;
  slew_lower_threshold_pct_fall : 50;
  slew_lower_threshold_pct_rise : 50;
  slew_upper_threshold_pct_fall : 50;
  slew_upper_threshold_pct_rise : 50;
  leakage_power_unit : "1pW";
  nom_process : 1.0;
  nom_temperature : 100.0;
  time_unit : "1ns";
  voltage_unit : "1v";
  default_input_pin_cap : 1;
  cell (DFF) {
    ff ("IQ", "IQN") {
      next_state : "D";
      clocked_on : "CLK";
    }
    area : 8;
    pin (D) {
      direction : "input";
      capacitance : 1;
      timing () {
        related_pin : "CLK";
        timing_type : hold_rising;
        fall_constraint (scalar) { values ("0"); }
        rise_constraint (scalar) { values ("0"); }
      }
      timing () {
        related_pin : "CLK";
        timing_type : setup_rising;
        fall_constraint (scalar) { values ("0"); }
        rise_constraint (scalar) { values ("0"); }
      }
    }
    pin (CLK) {
      direction : "input";
      clock : "true";
      capacitance : 1;
    }
	pin (Q) {
      direction : "output";
      function : "IQ";
      timing () {
        related_pin : "CLK";
        timing_type : "rising_edge";
        timing_sense : "non_unate";
        cell_fall (scalar) { values ("0"); }
        cell_rise (scalar) { values ("0"); }
        fall_transition (scalar) { values ("0"); }
        rise_transition (scalar) { values ("0"); }
      }
    }
  }
''')

cell_desc = [
	("ZERO", 0, "0", 1),
	("ONE", 0, "1", 1),
	("INV", 1, "!A", 2),
	("BUF", 1, "A", 4),
	("NAND", 2, "!(A * B)", 4),
	("NAND3", 3, "!(A * B * C)", 6),
	("NAND4", 4, "!(A * B * C * D)", 8),
	("NOR", 2, "!(A | B)", 4),
	("NOR3", 3, "!(A | B | C)", 6),
	("NOR4", 4, "!(A | B | C | D)", 8),
	("AND", 2, "A * B", 6),
	("AND3", 3, "A * B * C", 8),
	("AND4", 4, "A * B * C * D", 10),
	("OR", 2, "A | B", 6),
	("OR3", 3, "A | B | C", 8),
	("OR4", 4, "A | B | C | D", 10),
	("XOR", 2, "(A * !B) | (!A * B)", 12),
	("XNOR", 2, "(A * B) | (!A * !B)", 12),
	("AOI21", 3, "!(A * B | C)", 6),
	("OAI21", 3, "!((A | B) * C)", 6),
	("AO21", 3, "A * B | C", 8),
	("OA21", 3, "(A | B) * C", 8),
	("AOI22", 4, "!(A * B | C * D)", 8),
	("OAI22", 4, "!((A | B) * (C | D))", 8),
	("AO22", 4, "A * B | C * D", 10),
	("OA22", 4, "(A | B) * (C | D)", 10),
	("MAJI", 3, "!(A * B | B * C | A * C)", 12),
	("MAJ", 3, "(A * B | B * C | A * C)", 14),
	("MUXI", 3, "A & !B | !A * !C", 10),
	("MUX", 3, "A & B | !A * C", 12)
]

for (name, ninputs, function, ntransistors) in cell_desc:
	print(f"  cell ({name})" + " {")
	print(f"    area : {ntransistors + 1};")
	for _, input_name in zip(range(ninputs), "ABCD"):
		print(f"    pin ({input_name})" + " {")
		print("      capacitance : 1;")
		print("      direction : \"input\";")
		print("    }")
	print("    pin (Z) {")
	print(f"      function : \"{function}\";")
	print(f"      direction : \"output\";")
	for _, input_name in zip(range(ninputs), "ABCD"):
		print("      timing () {")
		print(f"        related_pin : \"{input_name}\";")
		print("        cell_rise (scalar) { values (\"1\"); }")
		print("        cell_fall (scalar) { values (\"1\"); }")
		print("        rise_transition (scalar) { values (\"1\"); }")
		print("        fall_transition (scalar) { values (\"1\"); }")
		print("      }")
	print("    }")
	print("  }")
print("}")

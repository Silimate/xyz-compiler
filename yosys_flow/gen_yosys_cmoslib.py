print('''library(yosys_cmoslib) {
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
  cell (_DFF_P_) {
    ff ("IQ", "IQN") {
      next_state : "D";
      clocked_on : "C";
    }
    area : 8;
    pin (D) {
      direction : "input";
      capacitance : 1;
      timing () {
        related_pin : "C";
        timing_type : hold_rising;
        fall_constraint (scalar) { values ("0"); }
        rise_constraint (scalar) { values ("0"); }
      }
      timing () {
        related_pin : "C";
        timing_type : setup_rising;
        fall_constraint (scalar) { values ("0"); }
        rise_constraint (scalar) { values ("0"); }
      }
    }
    pin (C) {
      direction : "input";
      clock : "true";
      capacitance : 1;
    }
    pin (Q) {
      direction : "output";
      function : "IQ";
      timing () {
        related_pin : "C";
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
	("_ZERO_", "", "0", 1),
	("_ONE_", "", "1", 1),
	("_NOT_", "A", "!A", 2),
	("_BUF_", "A", "A", 4),
	("_NAND_", "AB", "!(A * B)", 4),
	("_NOR_", "AB", "!(A | B)", 4),
	("_AND_", "AB", "A * B", 6),
	("_OR_", "AB", "A | B", 6),
	("_XOR_", "AB", "(A * !B) | (!A * B)", 9),
	("_XNOR_", "AB", "(A * B) | (!A * !B)", 9),
	("_ANDNOT_", "AB", "A * !B", 8),
	("_ORNOT_", "AB", "A | !B", 8),
	("_MUX_", "ABS", "!S * A | S * B", 12),
	("_NMUX_", "ABS", "!S * !A | S * !B", 10),
	("_AOI3_", "ABC", "!(A * B | C)", 6),
	("_OAI3_", "ABC", "!((A | B) * C)", 6),
	("_AOI4_", "ABCD", "!(A * B | C * D)", 8),
	("_OAI4_", "ABCD", "!((A | B) * (C | D))", 8),
]

for (name, inputs, function, ntransistors) in cell_desc:
	print(f"  cell ({name})" + " {")
	print(f"    area : {ntransistors + 1};")
	for _, input_name in enumerate(inputs):
		print(f"    pin ({input_name})" + " {")
		print("      capacitance : 1;")
		print("      direction : \"input\";")
		print("    }")
	print("    pin (Y) {")
	print(f"      function : \"{function}\";")
	print(f"      direction : \"output\";")
	for _, input_name in enumerate(inputs):
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

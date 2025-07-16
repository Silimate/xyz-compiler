# usage: yosys -c flow.tcl -- <path-to-rtlil-input>

yosys -import

proc xyz {} {
	log -header "Executing XYZ."
	log -push
	foreach module_name [split [string trim [yosys tee -q -s result.string select -list-mod]] "\n"] {
		log -header "Processing module $module_name."
		json -o /tmp/to-xyz.json $module_name
		yosys exec -expect-return 0 -- ../target/release/xyz-compiler \
			--no-lower-arith --no-sequential-mapping \
		 	/tmp/to-xyz.json yosys_cmos.lib /tmp/from-xyz.json
		delete $module_name
		read_json /tmp/from-xyz.json
		foreach {name} [list NOT BUF NAND NOR AND OR XOR XNOR ANDNOT ORNOT MUX NMUX AOI3 OAI3 AOI4 OAI4 DFF_P] {
			chtype -map "_${name}_" "\$_${name}_"
		}
		techmap -map tie_unmap.v
	}
	log -pop
}

read_rtlil "[lindex $argv 0]"
hierarchy -auto-top
synth -run :coarse
memory
demuxmap
bwmuxmap

xyz

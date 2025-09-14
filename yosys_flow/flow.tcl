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

set fn "[lindex $argv 0]"
if {[file extension $fn] == ".il"} {
	read_rtlil $fn	
} elseif {[file extension $fn] == ".v"} {
	read_verilog $fn
} elseif {[file extension $fn] == ".aig"} {
	read_aiger -module_name top $fn
} elseif {[file extension $fn] == ".json"} {
	read_json $fn
} else {
	puts "unknown format"
	exit 1
}

hierarchy -auto-top
prep -rdff
demuxmap
bwmuxmap

xyz

stat

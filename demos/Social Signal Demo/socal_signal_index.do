clear all
macro drop _all
frame create est
frame create temp  

/*	  
Goal: Produce firm-day level social signal indices between 2012 and 2021 

*/

use "reg_data_jfe",replace 

* grouping 
	local subsplit 1 // attnetion group defined within each year 
 
		
* rename 
	unab varlist: *num_mes*
	foreach var of local varlist{
		local vname=subinstr("`var'","num_mes","num",.)
		rename `var' `vname'
	}
	 	 
* sample selection   	
	keep if num>=10
		 
* re-do sentiment norm in-sample 	
	local sentiment sent_toppct sent_prof sent_interm sent_novice sent_noexp sent_self tw_sent sa_sent 
	foreach var of local sentiment{
		norm `var', method(zee)
		local label : variable label `var'
		label var zee_`var' "`label' (z)"
	}	
		
	unab sentiment: zee_*sent* 
	di "`sentiment'"
	 
* re-do attention norm in-sample 	
	local attention num_toppct num_prof num_interm num_novice num_noexp num_self tw_num sa_num 
 	foreach var of local attention{ 
		capture confirm var tmp_sum
		if _rc==0 drop tmp_sum
		bys date: egen tmp_sum=sum(`var')
		gen share_`var'=(`var'/tmp_sum)*100
		replace share_`var'=0 if `var'==0 & mi(share_`var')
 		
		local label : variable label `var' 
		label var share_`var' "`label'"
			
		norm share_`var', method(zee)
		label var zee_share_`var' "`label' (z)"
		drop share_`var' 
	}
	 
	unab attention: zee_share_*  	
	di "`attention'"
 	
* pc estimate	
	pca `sentiment'
	predict sent_pc, score
			
	pca `attention'
	predict attn_pc, score
		
	norm sent_pc, method(zee)
	label var zee_sent_pc "Sentiment PC1 (z)"
	 
	norm attn_pc, method(zee)
	label var zee_attn_pc "Attention PC1 (z)"	

	keep permno date sent_pc attn_pc zee_sent_pc zee_attn_pc
	gsort permno date
	compress 
	save "$dta/social_signal_index.dta",replace
 

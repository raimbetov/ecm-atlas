solve this task wiht multi agent

/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights

У меня в этой папке есть очень много мета-инсайтов, там можешь все их посмотреть. Твоя задача, проверить все инсайты на обновленном датасете, а обновленный датасет вот здесь. ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv

run for me 2 claude code agents and 1 codex agent in paralel. Look at multi agent framework(/Users/Kravtsovd/projects/chrome-extension-tcs/algorithms/product_div/Multi_agent_framework/00_MULTI_AGENT_ORCHESTRATOR.md) - but you ned to slitly change it - dont run gemini
but insted fo gemini run 2 claude code agents.
exa ple here /ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/run_2claude_1codex.sh 

then after we have all 3 agent done. compate results 


All docuemnta gent crate thye shoudl create in their folder wiht prefic agent (claude_1m,calude_2, codex) and they shoudl create their docuemtn in knwoladdge framework./ /Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md

BE sure to create  a new CSV aretefact in agent fodler!!!!! 

After all agents will finsih first comit push results. 

Tehn do complation, check resutls of all agents, and compare who didi it right.  

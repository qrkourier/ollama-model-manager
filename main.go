package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
)

const (
	modelsFile = "models.json"
	ollamaURL  = "http://localhost:11434/api/pull"
)

type SystemResources struct {
	HasNvidia bool
	VRAM_MB   int
	SysRAM_MB int
}

type SearchCriteria struct {
	MaxParameters int
	PreferredTags []string
	RequiredCaps  []string
	Constraint    string
}

type ModelEntry struct {
	Name         string   `json:"name"`
	Progress     float64  `json:"progress"` // 0.0 to 100.0
	Reasoning    string   `json:"reasoning"`
	OptimalTasks []string `json:"optimal_tasks"`
	Parameters   string   `json:"parameters"`
}

type PullResponse struct {
	Status    string `json:"status"`
	Total     int64  `json:"total"`
	Completed int64  `json:"completed"`
}

type HFModel struct {
	ID        string `json:"id"`
	Downloads int    `json:"downloads"`
}

type pendingModel struct {
	idx  int
	size float64
	name string
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		return
	}

	command := os.Args[1]
	switch command {
	case "pull":
		pullModels()
	case "describe":
		describeModels()
	case "discover":
		discoverModels()
	default:
		fmt.Printf("Unknown command: %s\n", command)
		printUsage()
	}
}

func printUsage() {
	fmt.Println("Usage: omm <command>")
	fmt.Println("\nCommands:")
	fmt.Println("  pull      Download pending models (optimized for speed & concurrency)")
	fmt.Println("  describe  Display a table of managed models and optimal use cases")
	fmt.Println("  discover  Evaluate local hardware and discover optimal models from Hugging Face")
}

func getSystemResources() SystemResources {
	res := SystemResources{}

	if memBytes, err := os.ReadFile("/proc/meminfo"); err == nil {
		lines := strings.Split(string(memBytes), "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					if kb, err := strconv.Atoi(fields[1]); err == nil {
						res.SysRAM_MB = kb / 1024
					}
				}
				break
			}
		}
	}

	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
	if out, err := cmd.Output(); err == nil {
		vramStr := strings.TrimSpace(string(out))
		vrams := strings.Split(vramStr, "\n")
		if len(vrams) > 0 {
			if vramMB, err := strconv.Atoi(strings.TrimSpace(vrams[0])); err == nil {
				res.HasNvidia = true
				res.VRAM_MB = vramMB
			}
		}
	}

	return res
}

func evaluateCriteria(res SystemResources) SearchCriteria {
	crit := SearchCriteria{
		PreferredTags: []string{"instruct", "coder", "base"},
		RequiredCaps:  []string{"bash", "linux", "sysadmin", "networking"},
	}

	if res.HasNvidia {
		usableMB := float64(res.VRAM_MB) * 0.9
		crit.MaxParameters = int(usableMB / 700.0)
		crit.Constraint = fmt.Sprintf("NVIDIA Dedicated GPU (%d MB VRAM) - STRICT VRAM FIT", res.VRAM_MB)
	} else {
		if res.SysRAM_MB > 32000 {
			crit.MaxParameters = 14
			crit.Constraint = "iGPU / High System RAM (>32GB) - Bandwidth Bound"
		} else {
			crit.MaxParameters = 8
			crit.Constraint = "iGPU / Low System RAM - Bandwidth Bound"
		}
	}

	return crit
}

func readModels() ([]ModelEntry, error) {
	file, err := os.Open(modelsFile)
	if err != nil {
		if os.IsNotExist(err) {
			return initializeDefaultModels(), nil
		}
		return nil, err
	}
	defer file.Close()

	var entries []ModelEntry
	if err := json.NewDecoder(file).Decode(&entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func saveModels(models []ModelEntry) {
	data, err := json.MarshalIndent(models, "", "  ")
	if err != nil {
		fmt.Printf("Error encoding models: %v\n", err)
		return
	}
	os.WriteFile(modelsFile, data, 0644)
}

func parseSize(p string) float64 {
	p = strings.ToUpper(strings.TrimSpace(p))
	p = strings.TrimSuffix(p, "B")
	v, err := strconv.ParseFloat(p, 64)
	if err != nil {
		return 999.0
	}
	return v
}

func pullModels() {
	models, err := readModels()
	if err != nil {
		fmt.Printf("Error reading models: %v\n", err)
		return
	}

	hasReady := false
	var pending []pendingModel

	for i, m := range models {
		if m.Progress >= 100.0 {
			hasReady = true
		} else {
			pending = append(pending, pendingModel{idx: i, size: parseSize(m.Parameters), name: m.Name})
		}
	}

	if len(pending) == 0 {
		fmt.Println("All models are already downloaded.")
		return
	}

	// Sort pending models by size ascending
	sort.Slice(pending, func(i, j int) bool {
		return pending[i].size < pending[j].size
	})

	// Optimization A: If no models are ready, pull the fastest (smallest) one immediately
	if !hasReady {
		first := pending[0]
		fmt.Printf("\n[Optimization] No models ready. Prioritizing fastest download to unblock usage: %s\n", first.name)
		err := pullModel(first.name, first.idx, models, true)
		if err != nil {
			fmt.Printf("Failed to pull %s: %v\n", first.name, err)
		}
		pending = pending[1:]
	}

	if len(pending) == 0 {
		return
	}

	// Optimization B: Group similarly sized models and download concurrently
	// Largest models naturally fall to the end of the groups list.
	var groups [][]pendingModel
	currentGroup := []pendingModel{pending[0]}

	for i := 1; i < len(pending); i++ {
		// Group models within 3B parameters of each other
		if math.Abs(pending[i].size-currentGroup[0].size) <= 3.0 {
			currentGroup = append(currentGroup, pending[i])
		} else {
			groups = append(groups, currentGroup)
			currentGroup = []pendingModel{pending[i]}
		}
	}
	if len(currentGroup) > 0 {
		groups = append(groups, currentGroup)
	}

	// Dynamic Parallelism: Adjust concurrency based on group size and observed grouping.
	for _, group := range groups {
		fmt.Printf("\n[Optimization] Concurrently pulling %d similarly-sized models (~%.1fB params)\n", len(group), group[0].size)

		var wg sync.WaitGroup
		// Adjust parallelism based on group constraint (up to 3 to prevent disk IO locking)
		parallelism := len(group)
		if parallelism > 3 {
			parallelism = 3
		}
		sem := make(chan struct{}, parallelism)

		for _, pm := range group {
			wg.Add(1)
			go func(p pendingModel) {
				defer wg.Done()
				sem <- struct{}{}
				err := pullModel(p.name, p.idx, models, false)
				if err != nil {
					fmt.Printf("Failed to pull %s: %v\n", p.name, err)
				}
				<-sem
			}(pm)
		}
		wg.Wait()
	}
}

func updateProgress(idx int, progress float64, models []ModelEntry) {
	models[idx].Progress = progress
	saveModels(models)
}

func pullModel(name string, idx int, models []ModelEntry, isSequential bool) error {
	payload := map[string]string{"name": name}
	jsonData, _ := json.Marshal(payload)

	resp, err := http.Post(ollamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	lastPrinted := -1
	for {
		var pullResp PullResponse
		if err := decoder.Decode(&pullResp); err == io.EOF {
			break
		} else if err != nil {
			return err
		}

		if pullResp.Total > 0 {
			percent := (float64(pullResp.Completed) / float64(pullResp.Total)) * 100
			updateProgress(idx, percent, models)

			if isSequential {
				fmt.Printf("\rPulling %s: %.2f%%", name, percent)
			} else {
				intPct := int(percent)
				// Print every 10% to avoid garbling concurrent output
				if intPct%10 == 0 && intPct != lastPrinted {
					fmt.Printf("Pulling %s: %d%%\n", name, intPct)
					lastPrinted = intPct
				}
			}
		}
	}
	if isSequential {
		fmt.Println()
	}
	updateProgress(idx, 100.0, models)
	if !isSequential {
		fmt.Printf("Successfully pulled %s\n", name)
	}
	return nil
}

func describeModels() {
	models, err := readModels()
	if err != nil {
		fmt.Printf("Error reading models: %v\n", err)
		return
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "MODEL\tSIZE\tOPTIMAL TASKS\tREASONING")
	fmt.Fprintln(w, "-----\t----\t-------------\t---------")

	for _, m := range models {
		displayStr := m.Name
		if m.Progress < 100.0 {
			displayStr = fmt.Sprintf("%s %.0f%%", m.Name, m.Progress)
		}

		tasks := strings.Join(m.OptimalTasks, ", ")
		reason := m.Reasoning
		if len(reason) > 50 {
			reason = reason[:47] + "..."
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", displayStr, m.Parameters, tasks, reason)
	}
	w.Flush()
}

func discoverModels() {
	fmt.Println("Evaluating Linux system resources...")
	res := getSystemResources()

	fmt.Printf("Hardware Detected:\n")
	fmt.Printf("  - System RAM: %d MB\n", res.SysRAM_MB)
	if res.HasNvidia {
		fmt.Printf("  - NVIDIA GPU: Detected (%d MB VRAM)\n", res.VRAM_MB)
	} else {
		fmt.Printf("  - NVIDIA GPU: Not Detected (Relying on iGPU/CPU RAM)\n")
	}

	crit := evaluateCriteria(res)
	fmt.Printf("\nGenerated Search Criteria:\n")
	fmt.Printf("  - Constraint applied: %s\n", crit.Constraint)
	fmt.Printf("  - Max Model Size (Params): %dB\n", crit.MaxParameters)
	fmt.Printf("  - Required Capabilities: %v\n", crit.RequiredCaps)

	fmt.Printf("\nQuerying Hugging Face API for top GGUF coding models...\n")

	hfURL := "https://huggingface.co/api/models?search=gguf+coder&limit=5&sort=downloads"
	resp, err := http.Get(hfURL)
	if err != nil {
		fmt.Printf("Error contacting Hugging Face API: %v\n", err)
		return
	}
	defer resp.Body.Close()

	var hfModels []HFModel
	if err := json.NewDecoder(resp.Body).Decode(&hfModels); err != nil {
		fmt.Printf("Error decoding HF response: %v\n", err)
		return
	}

	fmt.Println("\nRecommended Models (Ollama compatible via hf.co/):")
	fmt.Println("--------------------------------------------------")
	for _, m := range hfModels {
		fmt.Printf("  hf.co/%s (Downloads: %d)\n", m.ID, m.Downloads)
	}
	fmt.Println("\nTo pull one of these models: omm pull (or use ollama CLI directly)")
}

func initializeDefaultModels() []ModelEntry {
	return []ModelEntry{
		{
			Name:         "qwen2.5-coder:7b",
			Progress:     100.0,
			Parameters:   "7B",
			OptimalTasks: []string{"Bash Scripting", "Sysadmin"},
			Reasoning:    "State-of-the-art coding model at 7B. Highly efficient for single-shot scripting.",
		},
		{
			Name:         "llama3.1:8b",
			Progress:     100.0,
			Parameters:   "8B",
			OptimalTasks: []string{"General Architecture", "Reasoning"},
			Reasoning:    "Gold standard for instruction following. High TPS on 780M with solid logic.",
		},
		{
			Name:         "mistral-nemo:12b",
			Progress:     100.0,
			Parameters:   "12B",
			OptimalTasks: []string{"Network Eng.", "Large Context"},
			Reasoning:    "128k context window + excellent multilingual/technical reasoning.",
		},
		{
			Name:         "qwen2.5:14b",
			Progress:     100.0,
			Parameters:   "14B",
			OptimalTasks: []string{"Complex Planning", "Security Rules"},
			Reasoning:    "Pushes the 780M bandwidth limit but offers superior architectural logic.",
		},
		{
			Name:         "deepseek-coder-v2:16b",
			Progress:     23.4,
			Parameters:   "16B",
			OptimalTasks: []string{"Deep Audit", "Complex Scripting"},
			Reasoning:    "Exceptional MoE coding model. May push bandwidth limits, testing required.",
		},
	}
}

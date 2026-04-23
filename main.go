package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
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
	fmt.Println("  pull      Download pending models")
	fmt.Println("  describe  Display a table of managed models and optimal use cases")
	fmt.Println("  discover  Evaluate local hardware and discover optimal models from Hugging Face")
}

// getSystemResources strictly evaluates Linux system resources
func getSystemResources() SystemResources {
	res := SystemResources{}

	// 1. Check System RAM (Linux)
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

	// 2. Check for NVIDIA GPU and VRAM
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

func pullModels() {
	models, err := readModels()
	if err != nil {
		fmt.Printf("Error reading models: %v\n", err)
		return
	}

	var wg sync.WaitGroup
	sem := make(chan struct{}, 2)

	for i := range models {
		if models[i].Progress >= 100.0 {
			continue
		}

		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("Starting pull for %s...\n", models[idx].Name)
			err := pullModel(models[idx].Name, idx, models)
			if err != nil {
				fmt.Printf("Failed to pull %s: %v\n", models[idx].Name, err)
			} else {
				fmt.Printf("Successfully pulled %s\n", models[idx].Name)
			}
		}(i)
	}

	wg.Wait()
}

func updateProgress(idx int, progress float64, models []ModelEntry) {
	models[idx].Progress = progress
	saveModels(models)
}

func pullModel(name string, idx int, models []ModelEntry) error {
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
			fmt.Printf("\rPulling %s: %.2f%%\n", name, percent)
		}
	}
	updateProgress(idx, 100.0, models)
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

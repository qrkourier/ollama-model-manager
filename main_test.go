package main

import (
	"testing"
)

func TestEvaluateCriteria(t *testing.T) {
	// Test Case 1: High VRAM NVIDIA GPU (24GB)
	resNvidiaHigh := SystemResources{HasNvidia: true, VRAM_MB: 24576, SysRAM_MB: 65536}
	crit1 := evaluateCriteria(resNvidiaHigh)
	if crit1.MaxParameters < 30 {
		t.Errorf("Expected MaxParameters to be around 31 for 24GB VRAM, got %d", crit1.MaxParameters)
	}

	// Test Case 2: Low VRAM NVIDIA GPU (8GB)
	resNvidiaLow := SystemResources{HasNvidia: true, VRAM_MB: 8192, SysRAM_MB: 16384}
	crit2 := evaluateCriteria(resNvidiaLow)
	if crit2.MaxParameters != 10 {
		t.Errorf("Expected MaxParameters to be 10 for 8GB VRAM, got %d", crit2.MaxParameters)
	}

	// Test Case 3: High System RAM (iGPU fallback)
	resSysHigh := SystemResources{HasNvidia: false, SysRAM_MB: 65536}
	crit3 := evaluateCriteria(resSysHigh)
	if crit3.MaxParameters != 14 {
		t.Errorf("Expected MaxParameters to be 14 for High SysRAM iGPU, got %d", crit3.MaxParameters)
	}

	// Test Case 4: Low System RAM (iGPU fallback)
	resSysLow := SystemResources{HasNvidia: false, SysRAM_MB: 16384}
	crit4 := evaluateCriteria(resSysLow)
	if crit4.MaxParameters != 8 {
		t.Errorf("Expected MaxParameters to be 8 for Low SysRAM iGPU, got %d", crit4.MaxParameters)
	}
}

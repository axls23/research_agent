package main

import (
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

// findPython returns the first available Python interpreter executable.
func findPython() string {
	candidates := []string{"python", "python3", "py"}
	for _, name := range candidates {
		if _, err := exec.LookPath(name); err == nil {
			return name
		}
	}
	return ""
}

func main() {
	exePath, err := os.Executable()
	if err != nil {
		log.Fatalf("cannot resolve launcher path: %v", err)
	}

	baseDir := filepath.Dir(exePath)
	script := filepath.Join(baseDir, "run_research_agent.py")

	if _, err := os.Stat(script); err != nil {
		log.Fatalf("expected to find run_research_agent.py next to the launcher: %v", err)
	}

	python := findPython()
	if python == "" {
		log.Fatal("python interpreter not found; install Python 3 to run the research agent")
	}

	cmd := exec.Command(python, script, "--mode", "langgraph", "--rigor", "prisma")
	cmd.Dir = baseDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(
		os.Environ(),
		"RESEARCH_AGENT_MODE=langgraph",
		"RESEARCH_AGENT_RIGOR=prisma",
	)

	log.Printf("Launching Research Agent via %s %s", python, script)
	if err := cmd.Run(); err != nil {
		log.Fatalf("research agent exited with error: %v", err)
	}
}

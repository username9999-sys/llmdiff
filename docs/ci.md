# CI integration

`llmdiff compare` exits with code **1** when the regression budget is exceeded,
so it can gate a pull request directly.

## GitHub Actions

```yaml
- name: LLM regression gate
  run: |
    pip install llmdiff
    llmdiff compare corpus.jsonl --format github --max-loss-rate 0.05
```

With a PR comment reporter:

```yaml
- name: LLM regression gate
  run: llmdiff compare corpus.jsonl --format github --output pr_comment.md --max-loss-rate 0.05
- if: failure()
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      await github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: fs.readFileSync('pr_comment.md', 'utf8'),
      });
```

## Other CI systems

Any system that fails on non-zero exit works:

```bash
llmdiff compare corpus.jsonl --format junit --output results.xml --max-loss-rate 0.05
# results.xml is consumable by Jenkins/GitLab test reports
```

SARIF output (`--format sarif`) integrates with GitHub code scanning.

## Gate options

| option | meaning |
|---|---|
| `--max-loss-rate R` | fail when overall loss rate > R |
| `--max-regressed-dims N` | fail when more than N dimensions regressed |
| `--min-win-rate R` | fail when overall win rate < R |
| `--dimensions a,b` | only evaluate these dimensions |
| `--tolerance T` | score difference treated as a tie |

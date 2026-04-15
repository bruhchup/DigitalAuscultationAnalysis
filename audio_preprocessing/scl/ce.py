import torch


def train_epoch(model, train_loader, train_transform, criterion, optimizer, scheduler, device, num_classes=4):
    TP = [0] * num_classes
    GT = [0] * num_classes
    epoch_loss = 0.0

    model.train()

    for data, target, _ in train_loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            data_t = train_transform(data)

        optimizer.zero_grad()
        output = model(data_t)
        loss = criterion(output, target)
        epoch_loss += loss.item()

        _, labels_predicted = torch.max(output, dim=1)
        for idx in range(num_classes):
            TP[idx] += torch.logical_and((labels_predicted == idx), (target == idx)).sum().item()
            GT[idx] += (target == idx).sum().item()

        loss.backward()
        optimizer.step()

    scheduler.step()
    epoch_loss = epoch_loss / len(train_loader)
    se = sum(TP[1:]) / sum(GT[1:])
    sp = TP[0] / GT[0]
    icbhi_score = (se + sp) / 2
    acc = sum(TP) / sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc


def val_epoch(model, val_loader, val_transform, criterion, device, num_classes=4):
    TP = [0] * num_classes
    GT = [0] * num_classes
    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():
        for data, target, _ in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(val_transform(data))
            loss = criterion(output, target)
            epoch_loss += loss.item()

            _, labels_predicted = torch.max(output, dim=1)
            for idx in range(num_classes):
                TP[idx] += torch.logical_and((labels_predicted == idx), (target == idx)).sum().item()
                GT[idx] += (target == idx).sum().item()

    epoch_loss = epoch_loss / len(val_loader)
    se = sum(TP[1:]) / sum(GT[1:])
    sp = TP[0] / GT[0]
    icbhi_score = (se + sp) / 2
    acc = sum(TP) / sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc


def train_ce(model, train_loader, val_loader, train_transform, val_transform, criterion, optimizer, epochs, scheduler, device, num_classes=4):
    train_losses = []
    val_losses = []
    best_icbhi_score = 0
    best_se = 0
    best_sp = 0
    best_epoch_icbhi = 0
    best_state = None

    for i in range(1, epochs + 1):
        print(f"Epoch {i}")

        train_loss, train_se, train_sp, train_icbhi_score, train_acc = train_epoch(
            model, train_loader, train_transform, criterion, optimizer, scheduler, device, num_classes)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.4f}\tSE: {train_se:.4f}\tSP: {train_sp:.4f}\tScore: {train_icbhi_score:.4f}\tAcc: {train_acc:.4f}")

        val_loss, val_se, val_sp, val_icbhi_score, val_acc = val_epoch(
            model, val_loader, val_transform, criterion, device, num_classes)
        val_losses.append(val_loss)
        print(f"Val loss: {val_loss:.4f}\tSE: {val_se:.4f}\tSP: {val_sp:.4f}\tScore: {val_icbhi_score:.4f}\tAcc: {val_acc:.4f}")

        if i == 1 or val_icbhi_score > best_icbhi_score:
            best_epoch_icbhi = i
            best_icbhi_score = val_icbhi_score
            best_se = val_se
            best_sp = val_sp
            best_state = {"encoder": model.state_dict()}

    print(f"Best ICBHI score: {best_icbhi_score:.4f} (SE:{best_se:.4f} SP:{best_sp:.4f}) at epoch {best_epoch_icbhi}")

    return best_state, best_icbhi_score, best_se, best_sp
